import asyncio
import logging
import typing as t
from io import StringIO

import discord
from redbot.core import commands
from redbot.core.i18n import Translator

from .abc import MixinMeta
from .common.calls import create_memory_call
from .common.constants import REACT_SUMMARY_MESSAGE
from .common.utils import can_use, embed_to_content, is_question

log = logging.getLogger("red.vrt.assistant.listener")
_ = Translator("Assistant", __file__)


class AssistantListener(MixinMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.responding_to = set()
        self._typing_blocked_until = 0.0

    def _api_health(self):
        return self.bot.get_cog("NebulaApiHealth")

    async def _api_health_allow(self, action: str, **metadata) -> bool:
        api_health = self._api_health()
        if api_health is None:
            return True

        allow = getattr(api_health, "allow", None)
        if allow is None:
            return True

        try:
            return await allow(action, **metadata)
        except Exception:
            log.warning("NebulaApiHealth allow check failed for action=%s", action, exc_info=True)
            return True

    async def _api_health_record_error(self, action: str, error: BaseException, **metadata) -> None:
        api_health = self._api_health()
        if api_health is None:
            return

        record_error = getattr(api_health, "record_error", None)
        if record_error is None:
            return

        try:
            await record_error(action, error, **metadata)
        except Exception:
            log.warning("NebulaApiHealth error recording failed for action=%s", action, exc_info=True)

    async def _api_health_record_suppressed(self, action: str, **metadata) -> None:
        api_health = self._api_health()
        if api_health is None:
            return

        record_suppressed = getattr(api_health, "record_suppressed", None)
        if record_suppressed is None:
            return

        try:
            await record_suppressed(action, **metadata)
        except Exception:
            log.warning("NebulaApiHealth suppression recording failed for action=%s", action, exc_info=True)

    @commands.Cog.listener("on_message_without_command")
    async def handler(self, message: discord.Message):
        # If message object is None for some reason
        if not message:
            return
        if message.author.id in self.responding_to:
            return
        # If message wasn't sent in a guild
        if not message.guild:
            return
        # If message was from a bot
        if message.author.bot and not self.db.listen_to_bots:
            return
        # Ignore messages without content
        if not message.content:
            if not message.embeds:
                return
            # Replace message content with embed content
            embed_to_content(message)
        # Ignore if channel doesn't exist
        if not message.channel:
            return
        # Check if cog is disabled
        if await self.bot.cog_disabled_in_guild(self, message.guild):
            return
        # Check permissions
        if not message.channel.permissions_for(message.guild.me).send_messages:
            return
        if not message.channel.permissions_for(message.guild.me).embed_links:
            return

        conf = self.db.get_conf(message.guild)
        if not conf.enabled or (not conf.api_key and not self.db.endpoint_override):
            return

        channel = message.channel
        mention_ids = [m.id for m in message.mentions]
        bot_mentioned = self.bot.user.id in mention_ids

        ref: discord.Message = None
        # If bot wasnt mentioned in the message, see if someone replied to it
        if not bot_mentioned and hasattr(message, "reference") and message.reference:
            ref = message.reference.resolved
            if not isinstance(ref, discord.Message):
                try:
                    ref = await message.channel.fetch_message(message.reference.message_id)
                except discord.HTTPException:
                    pass

        if ref and ref.author.id == self.bot.user.id:
            bot_mentioned = True

        # Ignore common prefixes from other bots
        if message.content.startswith((",", ".", "+", "!", "-", "><", "?", "$", "%", "^", "&", "*", "_")):
            return
        if not await can_use(message, conf.blacklist, respond=False):
            return
        if len(message.content.strip()) < conf.min_length:
            return

        handle_message_kwargs = {
            "message": message,
            "question": message.content,
            "conf": conf,
            "listener": True,
        }

        # Return under the following conditions
        if (ref is not None and ref.author.id != self.bot.user.id) or (
            mention_ids and self.bot.user.id not in mention_ids
        ):
            # Do not respond to messages that are replies to other messages or mention someone else
            return

        conditions = [
            (channel.id != conf.channel_id and channel.id not in conf.listen_channels),
            (not bot_mentioned or not conf.mention_respond),
        ]
        check_auto_answer = [
            conf.auto_answer,
            channel.id not in conf.auto_answer_ignored_channels,
            getattr(channel, "category_id", 0) not in conf.auto_answer_ignored_channels,
            message.author.id not in conf.tutors,
            not any([role.id in conf.tutors for role in getattr(message.author, "roles", [])]),
        ]
        if all(check_auto_answer):
            if is_question(message.content):
                # Check if any embeddings match above the threshold
                embedding = await self.request_embedding(message.content, conf)
                related = await asyncio.to_thread(
                    conf.get_related_embeddings,
                    guild_id=message.guild.id,
                    query_embedding=embedding,
                    top_n_override=1,
                    relatedness_override=conf.auto_answer_threshold,
                )
                conditions.append(len(related) == 0)
                if len(related) > 0:
                    handle_message_kwargs["model_override"] = conf.auto_answer_model
                    handle_message_kwargs["auto_answer"] = True
        if all(conditions):
            # Message was not in the assistant channel and bot was not mentioned and auto answer is enabled and no related embeddings
            return

        conditions = [
            (channel.id == conf.channel_id or channel.id in conf.listen_channels),
            not message.content.endswith("?"),
            conf.endswith_questionmark,
            self.bot.user.id not in mention_ids,
        ]
        if all(conditions):
            # Message was in the assistant channel and didn't end with a question mark while the config requires it
            return

        self.responding_to.add(message.author.id)
        try:
            now = asyncio.get_running_loop().time()
            typing_blocked_until = getattr(self, "_typing_blocked_until", 0)

            should_type = (
                not handle_message_kwargs.get("auto_answer", False)
                and now >= typing_blocked_until
            )

            if should_type:
                try:
                    async with channel.typing():
                        await self.handle_message(**handle_message_kwargs)
                except discord.HTTPException as e:
                    if e.status == 429:
                        self._typing_blocked_until = asyncio.get_running_loop().time() + 1800

                        log.warning(
                            "Typing indicator rate limited, disabling listener typing for 30 minutes. "
                            "guild_id=%s channel_id=%s user_id=%s code=%s",
                            message.guild.id,
                            channel.id,
                            message.author.id,
                            getattr(e, "code", None),
                        )
                        await self.handle_message(**handle_message_kwargs)
                    else:
                        raise
            else:
                await self.handle_message(**handle_message_kwargs)
        finally:
            self.responding_to.discard(message.author.id)

    @commands.Cog.listener("on_guild_remove")
    async def cleanup(self, guild: discord.Guild):
        if guild.id in self.db.configs:
            log.info(f"Bot removed from {guild.name}, cleaning up...")
            del self.db.configs[guild.id]
            await self.save_conf()

    @commands.Cog.listener("on_raw_reaction_add")
    async def remember(self, payload: discord.RawReactionActionEvent):
        """Save messages as embeddings when reacted to with :brain: emoji"""
        emoji = str(payload.emoji)
        if emoji != "\N{BRAIN}":
            return
        if payload.user_id == self.bot.user.id:
            return
        if not payload.guild_id:
            return
        guild = self.bot.get_guild(payload.guild_id)
        if not guild:
            return
        user = payload.member
        if not user:
            return
        # Ignore reactions added by other bots
        if user.bot:
            return
        channel = guild.get_channel(payload.channel_id)
        if not channel:
            return

        try:
            message = await channel.fetch_message(payload.message_id)
        except discord.HTTPException:
            return

        if not message:
            return
        if not message.content:
            return
        conf = self.db.get_conf(guild)
        if not conf.enabled:
            return
        if not conf.api_key and not self.db.endpoint_override:
            return
        # Check if cog is disabled
        if await self.bot.cog_disabled_in_guild(self, guild):
            return
        if not any([role.id in conf.tutors for role in user.roles]) and user.id not in conf.tutors:
            return

        messages: t.List[discord.Message] = [message]

        # Set up the message chain
        tmp = message
        while True:
            if getattr(tmp, "reference", None) is not None:
                resolved = tmp.reference.resolved
                if resolved is None:
                    break
                messages.append(resolved)
                # If the message is a reply to another message, we want to keep going up the chain
                tmp = resolved

            else:
                break

        messages.reverse()

        content = StringIO()
        for idx, msg in enumerate(messages):
            if idx == 0:
                content.write(f"Original message from {msg.author.name}: {msg.content}\n")
            else:
                content.write(f"{msg.author.name} said: {msg.content}\n")

        success = True
        try:
            messages = [
                {"role": "developer", "content": REACT_SUMMARY_MESSAGE.strip()},
                {"role": "user", "content": content.getvalue()},
            ]
            res = await create_memory_call(messages=messages, api_key=conf.api_key, base_url=self.db.endpoint_override)
            if res:
                embedding = await self.add_embedding(guild, res.memory_name, res.memory_content)
                if embedding is None:
                    success = False
                else:
                    log.info(
                        "Created embedding in %s\nName: %s\nEntry: %s",
                        guild.name,
                        res.memory_name,
                        res.memory_content,
                    )
            else:
                success = False
        except Exception:
            log.warning("Failed to save embed memory in %s", guild.name, exc_info=True)
            success = False

        try:
            if success:
                await message.add_reaction("\N{WHITE HEAVY CHECK MARK}")
            else:
                await message.add_reaction("\N{CROSS MARK}")
        except discord.HTTPException:
            log.warning(
                "Failed to add memory reaction. guild_id=%s channel_id=%s message_id=%s",
                guild.id,
                channel.id,
                message.id,
            )
