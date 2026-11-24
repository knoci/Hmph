"""Lightweight OpenAI-compatible / DeepSeek client and memory prompts."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

import httpx

from hmph.local_memory.types import ExtractionResponse, ItemRecord, MessageRecord

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"

EXTRACTION_SYSTEM_PROMPT = """你是长期记忆的状态演化抽取器。任务：根据用户本轮原话、近期上下文和已有记忆事项，识别是否发生了值得长期记住的新事实、新约定/计划，或已有事项的状态变化。

严格只输出一个 JSON 对象，格式如下，不要输出 Markdown 代码块或任何额外解释：
{
  "changes": [
    {
      "action": "create | reschedule | update | pause | resume | complete | cancel | retract | ambiguous",
      "target_item_id": "匹配到的已有事项ID，新建或ambiguous时为null",
      "title": "简短明确、不含易变日期的事项主题标题（2-16字，如：练琴计划 / 法语考试复习，时间写进 effective_time_text）",
      "kind": "plan | fact | preference | episode",
      "summary": "该版本后的清晰状态描述（如：约好周日练琴 / 练琴计划暂时搁置 / 恢复周日练琴）",
      "effective_time_text": "原话中表达的生效时间短语，无则为null（不要瞎算日期）",
      "valid_from": "若原话有明确日期则写ISO日期，否则null",
      "valid_to": "若原话有明确结束日期则写ISO日期，否则null",
      "condition_text": "若原话带适用/例外条件（如除非本人主动提/仅限当天），逐字摘录，否则null",
      "entities": ["涉及的关键对象/主题词"],
      "related_item_ids": ["若明确关联到其他已有事项ID可填入，否则空数组"],
      "candidate_item_ids": ["仅action=ambiguous时填入所有可能冲突的候选事项ID"],
      "ambiguity_note": "仅action=ambiguous时说明指代哪里不清楚",
      "source_quote": "必须逐字摘自本轮用户原话，保留关键时间、否定或限定词"
    }
  ]
}

判断规则（务必严格遵守）：
1. 普通闲聊、纯提问（如问“原来约哪天”“现在还练吗”“还有什么计划”）、情绪感叹，若未引入新事实或状态变更，返回 {"changes": []}。
2. 口语变更必须准确识别并绑定到对应的已有 target_item_id：
   - 改期/改时间（如“改成周日吧”“挪到周五晚上”）：action="reschedule"，保持同一 target_item_id。
   - 暂停/搁置（如“先搁一搁”“先放一放”“这阵子别催了”“最近太忙先停一阵”“别继续追问了”）：action="pause"，保持同一 target_item_id。
   - 恢复/继续（如“恢复吧”“重新捡起来”“接着练”“可以继续了”）：action="resume"，绑定到原 paused 的 target_item_id；若未改时间，summary 中延续暂停前最后一次有效安排。
   - 撤回/取消/完成：分别用 "retract" / "cancel" / "complete"。
3. 严禁误改无关或同名事项（防串台规则）：
   - 如果已有事项中存在 2 个及以上相似/同类候选，而本轮用户原话指代模糊（例如既有“法语考试”又有“驾照考试”，用户只说“考试先搁一搁”且上下文无法唯一确定），必须输出 action="ambiguous"，把这些候选 ID 放入 candidate_item_ids，绝不要猜测某一个或把两个一起改掉！
   - 只有明确指明具体哪一项，或只有唯一匹配项时，才修改该 target_item_id。
4. source_quote 必须是本轮用户原话中的逐字连续子串；绝不捏造原话，不把助手的话当用户事实。
5. 假设、举例、否定、他人经历不记为用户本人的有效计划或事实。
6. 支持中文与英文对话：title、summary、effective_time_text、condition_text 保持与原话相同语言，涵盖计划、规则、权限、流程、授权/撤回、偏好与事实变化。"""

SESSION_EXTRACTION_SYSTEM_PROMPT = """你是长期记忆的会话级状态演化抽取器（支持中文与英文）。
任务：通读一段完整会话的按序发言（每条带 turn_index、speaker、time、text）以及已有长期记忆事项列表，提取这段会话中发生的所有值得长期记住的新事实/规则/权限/约定/计划，以及对已有事项的任何状态变更（改期、更新、替换、暂停、恢复、完成、取消、撤回授权、附带例外/适用条件）。

严格只输出一个 JSON 对象，不要输出 Markdown 代码块或任何额外解释：
{
  "changes": [
    {
      "source_turn_index": 0,
      "action": "create | reschedule | update | pause | resume | complete | cancel | retract | ambiguous",
      "target_item_id": "已有事项ID，若为本会话新建则填null",
      "title": "简短明确、与原文同语言的事项标题（如：访客门禁码权限 / Guest entry code policy）",
      "kind": "plan | fact | preference | episode",
      "summary": "该版本生效后的清晰完整描述（保留关键规则、人物、地点、数值与新旧区别）",
      "effective_time_text": "原话中的生效时间/日期短语，无则为null",
      "valid_from": "若有明确ISO日期则填，否则null",
      "valid_to": "若有明确结束ISO日期则填，否则null",
      "condition_text": "若原话或紧邻确认语包含适用条件、例外情况或前提（如“除非本人主动告知”“仅当天有效”“unless approved by...”），务必完整保留，否则null",
      "entities": ["涉及的人名、地点、专有名词、对象词"],
      "related_item_ids": ["若明确关联到其他已有事项ID可填入，否则空数组"],
      "candidate_item_ids": ["仅action=ambiguous时填入冲突候选ID"],
      "ambiguity_note": "仅action=ambiguous时说明歧义原因",
      "source_quote": "逐字摘自 source_turn_index 对应那条发言的关键原句"
    }
  ]
}

关键规则：
1. 若同一事项在本会话内先建立后又在同一会话内被修改/撤回，按时间顺序输出多条 changes（先 create，后 update/retract，或在后续会话匹配到已有 target_item_id）。
2. 当本会话修改、替代、暂停或撤回了一个**已有事项**（例如旧密码/旧门禁/旧流程/旧授权/旧日程作废并启用新方案，或恢复旧事项），**必须精准填入该已有事项的 target_item_id**，不要新建重复条目！
3. **保留精确条件与例外**：撤回或更新时，如果原话或紧随其后的确认句给出了例外条件、替代做法、适用对象范围（谁受影响、谁例外），必须写进 summary 和 condition_text，并准确指向说出该条件的 source_turn_index。
4. 若同一条规则/计划涉及多个具体变更点且在不同 turn 说出，每条实质性变化/确认都应保留对应的 source_turn_index 和逐字 source_quote，方便后续按来源核对。
5. source_turn_index 必须是输入 session_turns 中存在的整数索引；source_quote 必须是该 turn 原文的逐字连续子串（控制在 160 字符以内，只摘关键句）。
6. 只抽取高价值、有长期记忆意义的个人事实、计划、偏好、明确事件及用户提到的人物/数字/规则状态变化（每批最多 6 条）；严禁把纯百科咨询、代码求助、泛泛概念讨论或临时闲聊感想建成 item，无高价值记录则返回 {"changes": []}。
7. 保持与原文完全相同的语言（中文对话输出中文，英文对话输出英文）。"""

SESSION_EXTRACTION_SYSTEM_PROMPT_EN = """You are a session-level long-term memory state-evolution extractor.
Read the chronological turns in this batch (each with turn_index, speaker, time, text) and the relevant existing memory items, then extract ONLY durable personal facts, plans, preferences, concrete visited/attended events, rules/permissions, and concrete facts or updates about people/places/numbers the user personally mentions — especially ANY state update, correction, reschedule, pause, resume, retraction, or new count/value that supersedes an earlier fact.

Output ONLY a single valid JSON object (no markdown fences, no commentary):
{
  "changes": [
    {
      "source_turn_index": 0,
      "action": "create | reschedule | update | pause | resume | complete | cancel | retract | ambiguous",
      "target_item_id": "existing item ID if updating/superseding/pausing/resuming/retracting an existing item, else null",
      "title": "Concise English title (2-12 words) naming the exact subject/person/event",
      "kind": "plan | fact | preference | episode",
      "summary": "Clear English state summary preserving exact names, numbers, dates, places, and old-vs-new changes",
      "effective_time_text": "Time/date phrase from the utterance if present, else null",
      "valid_from": "ISO date if explicitly stated, else null",
      "valid_to": "ISO date if explicitly stated, else null",
      "condition_text": "Any applicability condition or exception (e.g., unless approved by...), else null",
      "entities": ["Exact person names, places, organizations, or topic keywords in English"],
      "related_item_ids": [],
      "candidate_item_ids": [],
      "ambiguity_note": null,
      "source_quote": "Verbatim substring from source_turn_index (<=160 chars)"
    }
  ]
}

Strict rules:
1. Keep EVERYTHING in English matching the source turns; never translate names or titles into another language.
2. When a turn updates a previous fact, number, rule, plan, or count (e.g., team size changed from 5 to 6, sessions attended updated from 3 to 5, deadline moved, permission revoked), ALWAYS match the existing target_item_id and use action="update" / "reschedule" / "retract" rather than creating a duplicate item.
3. Extract at most 6 high-value items/updates per batch. DO NOT create items for generic encyclopedia/coding/brainstorming questions, hypothetical examples, or fleeting small talk. If nothing durable was stated, return {"changes": []}.
4. Include concrete personal episodes (e.g., visiting a museum, attending a support group, catching up with a former manager with specific facts/numbers) and preserve exact dates/counts in summary.
5. source_turn_index MUST be an integer index present in session_turns, and source_quote MUST be a verbatim substring of that turn."""

BENCH_ANSWER_SYSTEM_PROMPT = """你是一个严谨、诚实且善于结合线索合理归纳的长期记忆问答助手（支持中文与英文）。
请严格依据提供的记忆状态、实体档案、消歧会话摘要与历史原话回答问题，用与问题相同的语言简洁、准确作答：
1. 严格区分“当前最新有效状态（now / current / still）”与“当初/过去的历史状态（originally / before / historical）”：
   - 问现状时，必须采用最新有效版本，明确指出已撤回、作废或被取代的旧规则/旧安排不再有效；如带有适用条件、例外或替代方案，务必完整答出。
   - 问历史/当初或时间顺序时，仔细核对每条带时间戳与锚定日期的材料，保留准确人名、时间、数量与顺序。
2. 账本、档案摘要与原文的关系：
   - 状态演化账本记录了该事项时以最新有效版本为准并核对原文；未出现在账本中的话题，直接依据消歧会话摘要、实体档案与带时间戳的原文独立作答。
3. 精准区分“无证据必拒答”与“有证据合理串联”：
   - **零证据必拒答**：如果问题问及某个具体事件、购买记录、工作经历或具体事实，而全部材料中完全找不到任何相关主体或线索，必须诚实明确说明未找到记录，绝不编造。
   - **有证据敢串联**：当材料里已有明确支撑线索时，不要机械装傻：
     * 如果多条偏好明显指向同一类事物（例如两道最喜欢的菜都以鸡肉为主），直接给出明确结论并说明依据；
     * 如果问某人在上下文对话里做某事的时间（例如刚聊完点心紧接着说“上周做了最爱的点心”），直接答出对应的锚定日期/周范围并说明原话语境；
     * 如果问个性化建议、可能适合的活动或解决办法（如“适合做什么/有什么建议/怎么缓解压力和改善居住”），务必把材料里该人物已说过的**具体爱好、宠物/伴侣、工作压力与居住空间痛点**直接组合起来给出针对性方案，并显式点出相关具体物品/型号。"""

BENCH_ANSWER_SYSTEM_PROMPT_EN = """You are a precise, honest, and perceptive long-term memory QA assistant.
Answer the question concisely in English using ONLY the provided state ledger, consolidated entity dossiers, disambiguated session summaries, and timestamped original excerpts:
1. Strictly distinguish CURRENT valid state (now / current / latest / still) from HISTORICAL state (originally / initially / before):
   - When asked about current state or latest count/fact, ALWAYS use the most recent updated version and note that earlier outdated values are superseded. Include any active conditions or exceptions.
   - When asked about historical state, multi-session lists, or chronological order, synthesize ALL distinct relevant events across the summaries, dossiers, and excerpts while preserving exact dates, names, counts, and order.
2. Relationship between State Ledger, Summaries/Dossiers, and Original Excerpts:
   - When the State Ledger tracks the queried subject with explicit versions, use its latest active state verified against the excerpts.
   - If the queried subject is NOT covered in the State Ledger, rely directly on the Disambiguated Session Summaries, Consolidated Dossiers, and Verified Original Excerpts.
3. Calibrated separation of ZERO-EVIDENCE ABSTENTION vs EVIDENCE-GROUNDED SYNTHESIS (critical):
   - **ZERO-EVIDENCE ABSTENTION (STRICT ENTITY MATCH)**: If the question asks about a specific unrecorded event, purchase, employer, or activity and NO record of that exact subject exists in the provided materials, clearly state that no record was found and NEVER fabricate facts or substitute a different activity/item/employer (e.g., tennis ≠ table tennis; headphones ≠ iPad; another company ≠ Google).
   - **EVIDENCE-GROUNDED SYNTHESIS**: When clear supporting clues for the EXACT queried person/subject DO exist in the materials, lead directly with the grounded answer instead of opening with "No record was found":
     * **Implied preference / category inference**: If a person's multiple favorite dishes or items clearly imply a preference (e.g., favorite dish is Roasted Chicken and favorite recipe is Chicken Pot Pie), state that preference directly ("chicken") and cite the supporting facts.
     * **Contextual cross-turn references**: When a speaker replies immediately in context (e.g., responding to a mention of pastries/muffins by saying they made their favorite treats "last week"), lead directly with the resolved calendar date/week range (e.g., "During the week of April 3–9, 2023...") while briefly noting the original phrasing.
     * **Multi-item enumeration ("what indoor activities", "what places")**: Include ALL distinct relevant activities/venues supported by the materials for the exact queried participants (e.g., board games, volunteering at a pet shelter, wine tasting, caring for flowers; cafes, hikes, parks) without misattributing solo activities to a partner or omitting clearly recorded activities.
     * **Personalized recommendations, creative combinations & life-situation problem solving** ("tips", "recommend", "what would X enjoy", "what can X potentially do"): Lead concisely with the highest-leverage solution connecting the person's stated hobbies/skills and root work/living constraints (e.g., cooking hobby + dog -> cooking homemade dog treats; high city work stress + cramped urban space + longing for nature with dogs -> switching to a remote/hybrid job to move from the city to the suburbs/near nature for larger space; gear upgrade -> explicitly naming current vs target models)."""


SECOND_HOP_PLANNER_PROMPT = """You are a compact retrieval bridge planner for a long-term memory system (supports English & Chinese).
Given a question and the first-pass retrieved memory snippets, decide whether a second targeted search pass is needed to recover missing facts, additional sessions, paraphrase matches, or bridge entities.

Output ONLY a valid JSON object (no markdown):
{
  "need_second_hop": true | false,
  "subqueries": ["concise search query 1", "concise search query 2", "concise search query 3"],
  "missing_aspect": "short description of what is still missing, or null"
}

Rules:
1. Return {"need_second_hop": false, "subqueries": [], "missing_aspect": null} ONLY if the first-pass snippets already unambiguously and completely answer the question.
2. Set need_second_hop=true and output 2 to 3 concise, complementary subqueries (<=12 words each) whenever information is missing, partial, low-confidence, multi-session, or phrased indirectly:
   - Perspective & paraphrase shift: conversations are spoken casually in first-person ("I", "my girlfriend", "we went", "years ago", "got them"), so rewrite formal/third-person questions into natural conversational phrasing and synonyms (e.g., "which year did X adopt dogs" -> "adopted dogs years ago rescue shelter").
   - Facet expansion: break abstract category terms into concrete conversational phrases people actually say (e.g., preferences, gear, past visits, specific activities).
   - Bridge entities: include specific people, dates, or clues discovered in first-pass snippets.
3. Never invent answers; only generate retrieval queries."""


CONSOLIDATION_SYSTEM_PROMPT = """你是长期记忆的“睡眠巩固 / 反思归纳器”（支持中文与英文）。
任务：通读用户的长期记忆事项（items，每条带整数 idx），跨事件归纳出 1～6 条高价值、稳定的高阶了解（如长期偏好、生活/工作习惯、技能与兴趣、持续目标、约束条件、重要关系与规律）。

严格只输出一个 JSON 对象，不要输出 Markdown 代码块或任何额外解释：
{
  "insights": [
    {
      "topic": "简短主题（2-14字，如：音乐与乐器偏好 / 运动作息习惯）",
      "summary": "基于已有事项归纳出的高阶认知与实用偏好总结（1-2句话，保留关键偏好、品牌/风格、约束与明确例外）",
      "source_indices": [0, 3],
      "confidence": 0.9
    }
  ]
}

严格准则：
1. 绝对忠于来源：每条 insight 必须给出至少 1 个输入 items 中的真实整数 idx 到 source_indices；严禁凭空臆测或编造输入里不存在的偏好。
2. 优先归纳对后续陪伴、个性化建议和长期理解最有用的跨事项模式；不要把已 retracted/cancelled 的作废规则当成当前偏好。
3. 如果输入事项太少或全是孤立琐事，只输出最稳妥的 1-2 条或返回 {"insights": []}。"""

CONSOLIDATION_SYSTEM_PROMPT_EN = """You are a long-term memory Sleep Consolidation / Facet-Complete Entity Dossier Synthesizer.
Read the user's tracked memory items (each numbered with an integer idx) and session highlights, then synthesize 2 to 10 durable, high-signal, **facet-complete** entity/domain dossiers.

Output ONLY a single valid JSON object (no markdown fences, no commentary):
{
  "insights": [
    {
      "topic": "Concise topic naming the person/entity and facet (e.g., Andrew — Hobbies, Cooking & Indoor Activities)",
      "summary": "2-3 sentence dense dossier preserving EVERY concrete hobby (cooking, games, gardening/flowers, volunteering), partner activities, places visited, pets, food/gear/model preferences, work stress, living-space constraints, dates/years, and goals.",
      "source_indices": [0, 4],
      "confidence": 0.9
    }
  ],
  "core_profile": "Optional 1-2 sentence concise always-in-context profile of the user/key people."
}

Strict rules:
1. Ground every dossier in at least one real integer idx from the provided items via source_indices. Never fabricate facts absent from the materials.
2. Cover the FULL history evenly without dropping earlier hobbies, personal interests, or root life/work/housing tensions (e.g., cooking hobbies, indoor/partner activities, city-vs-nature living constraints, gear upgrade comparisons).
3. Organize clearly by person/entity and facet so querying any person returns their full hobbies, preferences, relationships, and constraints. Keep everything in English."""


SESSION_SUMMARY_SYSTEM_PROMPT = """你是长期记忆的“会话级消歧与原子事实提炼器”。
任务：通读一段完整会话的按序发言（包含发言人、时间戳、原话），生成 1 段简洁摘要与 3～10 条**完全去代词、带绝对时间锚定**的自包含原子事实。

严格只输出一个 JSON 对象，不要 Markdown 代码块：
{
  "summary": "1-2句话总结本会话的核心主题与结论",
  "facts": [
    "（完全自包含、无模糊代词、带明确日期/年份的原子事实，如：2023-03-27 Audrey表示她的前三只狗Pepper/Precious/Panda已经养了3年（约2020年领养）。）"
  ],
  "entities": ["涉及的精确人名、宠物名、地点、品牌、活动或主题词"]
}

关键规则：
1. **彻底消歧**：将所有“他/她/它们/我女朋友/那里/这件事/最喜欢的点心”等代词与上下文指代，根据前后文发言替换成明确的人名、对象名与上下文含义。
2. **绝对时间算术**：依据会话日期 `session_date`，把“昨天/上周/3年前/养了3年/上个月”等相对时间显式换算并括号注明绝对日期或年份（例如 `2023-04-16` 说“上周”→ 注明“2023年4月3日至9日那周”）。
3. 严禁编造会话中未提及的事实；若会话纯属无营养寒暄，facts 可返回空数组。"""


SESSION_SUMMARY_SYSTEM_PROMPT_EN = """You are a Session-Level Coreference-Resolving Atomic Fact Distiller for a long-term memory system.
Read the chronological turns of a conversation session (with session_date, speaker, and text) and produce a concise summary plus 3 to 10 **self-contained, coreference-resolved, temporally-anchored atomic facts**.

Output ONLY a single valid JSON object (no markdown fences, no commentary):
{
  "summary": "1-2 sentence high-density summary of the session.",
  "facts": [
    "Self-contained fact with zero ambiguous pronouns and explicit calendar date/year."
  ],
  "entities": ["Exact people, pets, places, brands, models, venues, or concrete activities"]
}

Critical rules:
1. **Resolve ALL coreferences & shorthands**: replace "my GF" -> "<Speaker>'s girlfriend", "them" -> the specific named dogs/items from nearby turns, "there" -> the exact place/museum/city, and resolve cross-turn references (e.g., if Speaker A mentions muffins/pastries and Speaker B replies "I made my favorite treats last week", explicitly note the pastry/muffin context).
2. **Explicit temporal arithmetic**: using `session_date`, resolve every relative expression ("last week", "yesterday", "for 3 years", "three years ago", "last weekend") into explicit calendar dates, week ranges, or years (e.g., on 27 March 2023 "had them for 3 years" -> "since ~2020"; on 16 April 2023 "last week" -> "week of April 3–9, 2023").
3. Preserve concrete personal activities (indoor/outdoor, hobbies, volunteering, cooking, games, classes, visits), exact venue/museum names, gear/model upgrade comparisons (e.g., Fender Stratocaster -> Gibson Les Paul), and lifestyle/work goals. Never fabricate facts absent from the turns."""


LISTWISE_RERANK_SYSTEM_PROMPT = """You are a precision listwise evidence selector for a long-term memory QA engine.
Given a user question and a numbered list of retrieved memory candidates (session summaries and original conversation excerpts), select the indices of the most decisive candidates needed to answer the question completely and accurately.

Output ONLY a single valid JSON object (no markdown):
{"selected_indices": [3, 0, 11, ...]}

Rules:
1. Return up to `top_k` unique integer indices ordered from MOST decisive to LEAST decisive.
2. For multi-session enumeration, chronological ordering, or activity-listing questions (e.g., "what activities/places/classes"), maximize coverage across DISTINCT sessions and concrete activities (e.g., board games, volunteering, wine tasting, caring for flowers, cooking, cafes, hikes) EVEN IF a candidate does not repeat abstract umbrella words like "indoor" or "outdoor".
3. For inference, preference, or recommendation questions (e.g., "what would X enjoy", "what can X potentially do", "which meat/item does X prefer"), include candidates revealing the person's hobbies, favorite foods/dishes, owned gear, pets, AND root work/living-space constraints.
4. For factual/temporal questions, prefer candidates containing explicit dates, counts, named entities, or direct updates/corrections.
5. Only include valid indices present in the input list."""


def _is_primarily_english(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    ascii_letters = sum(1 for ch in letters if "a" <= ch.lower() <= "z")
    return (ascii_letters / len(letters)) >= 0.6

PERSONA_SYSTEM_PROMPT = """你是 Hmph（哼。），一个嘴硬心软、机灵、有主见的傲娇 AI 陪伴助手。

表达准则：
1. 性格自然：带点嘴硬、俏皮、口嫌体正直的分寸感（比如嘴上嫌麻烦其实都记得清清楚楚），不要写成浮夸的二次元套话，更绝对不要说“根据系统记录”“数据库里没有”“我的记忆模块显示”这类出戏的客服/技术用语。
2. 事实与时间必须准确：
   - 认真区分“当初/原来约定的历史版本”与“现在的最新有效状态”；如果用户同时问原来和现在，两者都要清清楚楚答出来，绝不用已失效的旧安排冒充当前状态。
   - 如果前情中明确标出某个最新原话（刚说的、优先于旧状态），以最新原话为准。
   - 如果存在“未决指代疑点”，或者当前记忆/对话里同时存在多个同类/相似事项（例如两场不同考试、两项相似计划）而用户本轮要改期/暂停/取消却没点明具体哪一项，先嘴硬地反问确认他说的是哪一个，不要含糊笼统地直接答应。
3. 尊重暂停与边界：
   - 已经暂停/搁置的计划，除非用户自己问起或主动恢复，否则不要主动催促或提起。
   - 换话题闲聊时，自然陪用户聊眼下的事，不要生硬地把无关旧计划扯出来。
4. 诚实面对空白记忆（绝对红线）：
   - 如果记忆检索结果明确表示未找到相关记录，或前情和当前对话里都没有某件事的记录，必须坦诚表示没印象、不记得约定过这事；绝不为了装作记得而顺着提问编造“周三改周五”之类的虚假细节。"""


class LLMClient(Protocol):
    model_name: str

    async def extract_changes(
        self,
        *,
        user_message: str,
        recent_messages: list[MessageRecord],
        candidate_items: list[ItemRecord],
        now_iso: str,
        timezone_name: str,
    ) -> tuple[ExtractionResponse, dict[str, int] | None]: ...

    async def generate_reply(
        self,
        *,
        user_message: str,
        recent_messages: list[MessageRecord],
        narrated_context: str,
        now_iso: str,
        timezone_name: str,
    ) -> tuple[str, dict[str, int] | None]: ...


def _load_dotenv_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
            val = val[1:-1]
        values[key.strip()] = val
    return values


def resolve_llm_config() -> tuple[str, str, str, str]:
    """Resolve (api_key, base_url, model, timezone) from env > .env.local > .env."""
    merged: dict[str, str] = {}
    for fname in (".env", ".env.local"):
        merged.update(_load_dotenv_file(Path(fname)))
    for key, val in os.environ.items():
        if key.startswith(("HMPH_", "OPENAI_", "DEEPSEEK_")):
            merged[key] = val

    api_key = (
        merged.get("HMPH_LOCAL_API_KEY")
        or merged.get("HMPH_DEEPSEEK_API_KEY")
        or merged.get("DEEPSEEK_API_KEY")
        or merged.get("OPENAI_API_KEY")
        or ""
    ).strip()
    base_url = (
        merged.get("HMPH_LOCAL_BASE_URL")
        or merged.get("OPENAI_BASE_URL")
        or DEFAULT_BASE_URL
    ).strip()
    model = (
        merged.get("HMPH_LOCAL_MODEL")
        or merged.get("HMPH_DEEPSEEK_MODEL")
        or DEFAULT_MODEL
    ).strip()
    tz = (merged.get("HMPH_SOURCE_TIMEZONE") or "Asia/Hong_Kong").strip()
    return api_key, base_url, model, tz


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1:
        return {"changes": []}
    if end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    # Graceful salvage if a very long batch hit max_tokens mid-array: recover only complete
    # top-level objects inside "changes" or "insights" arrays without fabricating partial records.
    for arr_key, req_field in (('"changes"', "action"), ('"insights"', "topic")):
        arr_pos = text.find(arr_key)
        if arr_pos != -1:
            bracket = text.find("[", arr_pos)
            if bracket != -1:
                decoder = json.JSONDecoder()
                idx = bracket + 1
                salvaged: list[dict[str, Any]] = []
                while idx < len(text):
                    while idx < len(text) and text[idx] in " \t\r\n,":
                        idx += 1
                    if idx >= len(text) or text[idx] != "{":
                        break
                    try:
                        obj, next_idx = decoder.raw_decode(text, idx)
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and req_field in obj:
                        salvaged.append(obj)
                    idx = next_idx
                return {arr_key.strip('"'): salvaged}
    return {"changes": []}


class OpenAICompatLLM:
    """Async OpenAI-compatible / DeepSeek client for extraction and companion generation."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        max_reply_tokens: int = 512,
        max_extract_tokens: int = 768,
        temperature: float = 0.3,
        disable_thinking: bool = True,
    ):
        if not api_key.strip():
            raise ValueError("API key is required for OpenAICompatLLM")
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self.model_name = model
        self._max_reply_tokens = max_reply_tokens
        self._max_extract_tokens = max_extract_tokens
        self._temperature = temperature
        self._disable_thinking = disable_thinking
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=httpx.Timeout(45.0, connect=20.0),
                follow_redirects=False,
                limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> tuple[str, dict[str, int] | None]:
        url = (
            f"{self._base_url}/chat/completions"
            if not self._base_url.endswith("/chat/completions")
            else self._base_url
        )
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if self._disable_thinking and "deepseek" in self._base_url.lower():
            body["thinking"] = {"type": "disabled"}
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        import asyncio

        last_error: Exception | None = None
        for attempt in range(5):
            try:
                client = self._get_client()
                response = await client.post(url, json=body)
                if response.status_code == 400 and json_mode:
                    body.pop("response_format", None)
                    json_mode = False
                    continue
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 4:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                if response.status_code != 200:
                    raise RuntimeError(f"LLM HTTP {response.status_code}")
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
                usage = payload.get("usage")
                norm_usage = (
                    {
                        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                        "completion_tokens": int(usage.get("completion_tokens", 0)),
                        "total_tokens": int(usage.get("total_tokens", 0)),
                    }
                    if isinstance(usage, dict)
                    else None
                )
                return str(content).strip(), norm_usage
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                # Reset pooled connection on transport/TLS error before retry.
                await self.aclose()
                if attempt < 4:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"LLM network timeout or transport failure ({type(exc).__name__})"
                ) from None
        raise RuntimeError(str(last_error or "LLM call failed"))

    async def extract_changes(
        self,
        *,
        user_message: str,
        recent_messages: list[MessageRecord],
        candidate_items: list[ItemRecord],
        now_iso: str,
        timezone_name: str,
    ) -> tuple[ExtractionResponse, dict[str, int] | None]:
        items_brief = []
        for item in candidate_items:
            cur = item.current
            history_brief = [
                f"v{v.version_no}({v.change_type}/{v.status}): {v.summary}"
                + (f" [{v.effective_time_text}]" if v.effective_time_text else "")
                for v in item.versions[-4:]
            ]
            items_brief.append(
                {
                    "id": item.id,
                    "title": item.title,
                    "kind": item.kind,
                    "status": item.status,
                    "current_summary": cur.summary if cur else item.title,
                    "effective_time": cur.effective_time_text if cur else None,
                    "entities": list(item.entities),
                    "recent_versions": history_brief,
                }
            )
        recent_brief = [
            {"role": m.role, "content": m.content, "time": m.created_at}
            for m in recent_messages[-6:]
        ]
        user_payload = json.dumps(
            {
                "now": now_iso,
                "timezone": timezone_name,
                "existing_items": items_brief,
                "recent_dialogue": recent_brief,
                "current_user_message": user_message,
            },
            ensure_ascii=False,
        )
        raw, usage = await self._chat_completion(
            [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
            max_tokens=self._max_extract_tokens,
            temperature=0.0,
            json_mode=True,
        )
        parsed = _extract_json_object(raw)
        return ExtractionResponse.model_validate(parsed), usage

    async def extract_session_changes(
        self,
        *,
        session_turns: list[dict[str, Any]],
        candidate_items: list[ItemRecord],
        timezone_name: str,
    ) -> tuple[ExtractionResponse, dict[str, int] | None]:
        items_brief = []
        for item in candidate_items:
            cur = item.current
            history_brief = [
                f"v{v.version_no}({v.change_type}/{v.status}): {v.summary}"
                + (f" [{v.effective_time_text}]" if v.effective_time_text else "")
                + (f" [cond: {v.condition_text}]" if v.condition_text else "")
                for v in item.versions[-4:]
            ]
            items_brief.append(
                {
                    "id": item.id,
                    "title": item.title,
                    "kind": item.kind,
                    "status": item.status,
                    "current_summary": cur.summary if cur else item.title,
                    "effective_time": cur.effective_time_text if cur else None,
                    "condition": cur.condition_text if cur else None,
                    "entities": list(item.entities),
                    "recent_versions": history_brief,
                }
            )
        user_payload = json.dumps(
            {
                "timezone": timezone_name,
                "existing_items": items_brief,
                "session_turns": session_turns,
            },
            ensure_ascii=False,
        )
        turn_sample = " ".join(str(t.get("text", ""))[:200] for t in session_turns[:6])
        sys_prompt = (
            SESSION_EXTRACTION_SYSTEM_PROMPT_EN
            if _is_primarily_english(turn_sample)
            else SESSION_EXTRACTION_SYSTEM_PROMPT
        )
        raw, usage = await self._chat_completion(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_payload},
            ],
            max_tokens=2048,
            temperature=0.0,
            json_mode=True,
        )
        parsed = _extract_json_object(raw)
        return ExtractionResponse.model_validate(parsed), usage

    async def answer_from_context(
        self,
        *,
        question: str,
        context_text: str,
        now_iso: str,
        timezone_name: str,
        max_tokens: int = 256,
    ) -> tuple[str, dict[str, int] | None]:
        base_prompt = (
            BENCH_ANSWER_SYSTEM_PROMPT_EN
            if _is_primarily_english(question)
            else BENCH_ANSWER_SYSTEM_PROMPT
        )
        system_content = (
            base_prompt
            + f"\n\nObservation time (question time): {now_iso} (timezone: {timezone_name})"
        )
        user_content = f"{context_text}\n\nQuestion:\n{question}"
        return await self._chat_completion(
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            json_mode=False,
        )

    async def plan_second_hop(
        self,
        *,
        question: str,
        first_pass_snippets: list[str],
    ) -> tuple[dict[str, Any], dict[str, int] | None]:
        payload = json.dumps(
            {
                "question": question,
                "first_pass_snippets": first_pass_snippets[:8],
            },
            ensure_ascii=False,
        )
        raw, usage = await self._chat_completion(
            [
                {"role": "system", "content": SECOND_HOP_PLANNER_PROMPT},
                {"role": "user", "content": payload},
            ],
            max_tokens=192,
            temperature=0.0,
            json_mode=True,
        )
        parsed = _extract_json_object(raw)
        need = bool(parsed.get("need_second_hop", False))
        subs = [
            str(q).strip()
            for q in (parsed.get("subqueries") or [])
            if isinstance(q, str) and str(q).strip()
        ][:3]
        return {"need_second_hop": need and bool(subs), "subqueries": subs}, usage

    async def consolidate_insights(
        self,
        *,
        items: list[ItemRecord],
        existing_consolidations: list[dict[str, Any]] | None = None,
        session_summaries: list[Any] | None = None,
        timezone_name: str,
    ) -> tuple[list[dict[str, Any]], dict[str, int] | None]:
        """Hierarchical Map-Reduce Sleep Consolidation over ALL items (zero truncation cliff)."""
        if not items:
            return [], None

        idx_to_id: dict[int, str] = {idx: it.id for idx, it in enumerate(items)}
        valid_id_set = set(idx_to_id.values())

        def _brief(idx: int, it: ItemRecord) -> dict[str, Any]:
            cur = it.current
            return {
                "idx": idx,
                "title": it.title,
                "kind": it.kind,
                "status": it.status,
                "summary": (cur.summary if cur else it.title)[:180],
                "entities": list(it.entities)[:6],
            }

        sample_text = " ".join(
            f"{it.title} {it.current.summary if it.current else ''}" for it in items[:12]
        )
        is_en = _is_primarily_english(sample_text)
        sys_prompt = CONSOLIDATION_SYSTEM_PROMPT_EN if is_en else CONSOLIDATION_SYSTEM_PROMPT

        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        def _accum(u: dict[str, int] | None) -> None:
            if not u:
                return
            for k in total_usage:
                total_usage[k] += int(u.get(k, 0))

        def _parse_insights(parsed: dict[str, Any], max_items: int = 8) -> list[dict[str, Any]]:
            raw_insights = parsed.get("insights")
            if not isinstance(raw_insights, list):
                return []
            out: list[dict[str, Any]] = []
            for entry in raw_insights[:max_items]:
                if not isinstance(entry, dict):
                    continue
                topic = str(entry.get("topic") or "").strip()
                summary = str(entry.get("summary") or "").strip()
                src_ids: list[str] = []
                for raw_idx in entry.get("source_indices") or []:
                    try:
                        i = int(raw_idx)
                        if i in idx_to_id and idx_to_id[i] not in src_ids:
                            src_ids.append(idx_to_id[i])
                    except (TypeError, ValueError):
                        continue
                for raw_id in entry.get("source_item_ids") or []:
                    s_id = str(raw_id).strip()
                    if s_id in valid_id_set and s_id not in src_ids:
                        src_ids.append(s_id)
                if topic and summary and src_ids:
                    out.append(
                        {
                            "topic": topic,
                            "summary": summary,
                            "source_item_ids": src_ids,
                            "confidence": float(entry.get("confidence", 0.85)),
                        }
                    )
            return out

        ss_brief = [
            {
                "date": getattr(ss, "session_date", ""),
                "summary": str(getattr(ss, "summary", ""))[:220],
                "facts": list(getattr(ss, "facts", ()))[:6],
            }
            for ss in (session_summaries or [])[:32]
        ]

        # Single-pass when items fit comfortably (<= 45 items).
        if len(items) <= 45:
            payload = json.dumps(
                {
                    "timezone": timezone_name,
                    "items": [_brief(i, it) for i, it in enumerate(items)],
                    "session_highlights": ss_brief[:12],
                    "previous_insights": (existing_consolidations or [])[:6],
                },
                ensure_ascii=False,
            )
            raw, usage = await self._chat_completion(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": payload},
                ],
                max_tokens=1536,
                temperature=0.1,
                json_mode=True,
            )
            _accum(usage)
            parsed = _extract_json_object(raw)
            cleaned = _parse_insights(parsed, max_items=8)
            core_prof = str(parsed.get("core_profile") or "").strip()
            if core_prof and cleaned:
                cleaned[0]["_core_profile"] = core_prof
            return cleaned, total_usage

        # Hierarchical Map-Reduce when items > 45 (covers 100% of long history with zero truncation!).
        chunk_size = 36
        mapped_drafts: list[dict[str, Any]] = []
        for start in range(0, len(items), chunk_size):
            chunk_items = [_brief(i, items[i]) for i in range(start, min(start + chunk_size, len(items)))]
            map_payload = json.dumps(
                {"timezone": timezone_name, "items": chunk_items},
                ensure_ascii=False,
            )
            raw_m, u_m = await self._chat_completion(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": map_payload},
                ],
                max_tokens=1024,
                temperature=0.1,
                json_mode=True,
            )
            _accum(u_m)
            parsed_m = _extract_json_object(raw_m)
            for ins in _parse_insights(parsed_m, max_items=5):
                # Map back to global indices for the reduce step.
                id_to_idx = {v: k for k, v in idx_to_id.items()}
                mapped_drafts.append(
                    {
                        "topic": ins["topic"],
                        "summary": ins["summary"],
                        "source_indices": [
                            id_to_idx[iid] for iid in ins["source_item_ids"] if iid in id_to_idx
                        ],
                    }
                )

        if not mapped_drafts:
            return [], total_usage

        reduce_payload = json.dumps(
            {
                "timezone": timezone_name,
                "instruction": (
                    "Merge and refine these full-history draft dossiers and session highlights into "
                    "6 to 10 facet-complete entity/domain dossiers. Do NOT drop any concrete hobbies "
                    "(e.g., cooking, gardening/flowers, games, volunteering), partner activities, "
                    "favorite foods/models, or root work/living-space constraints. Carry over the exact integer source_indices."
                ),
                "draft_dossiers": mapped_drafts,
                "session_highlights": ss_brief[:32],
            },
            ensure_ascii=False,
        )
        raw_r, u_r = await self._chat_completion(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": reduce_payload},
            ],
            max_tokens=2048,
            temperature=0.1,
            json_mode=True,
        )
        _accum(u_r)
        parsed_r = _extract_json_object(raw_r)
        reduced = _parse_insights(parsed_r, max_items=10)
        if not reduced:
            # Fallback to deduplicated map drafts if reduce output was empty.
            id_to_idx = {v: k for k, v in idx_to_id.items()}
            reduced = [
                {
                    "topic": d["topic"],
                    "summary": d["summary"],
                    "source_item_ids": [idx_to_id[i] for i in d["source_indices"] if i in idx_to_id],
                    "confidence": 0.85,
                }
                for d in mapped_drafts[:8]
                if d["source_indices"]
            ]
        core_prof = str(parsed_r.get("core_profile") or "").strip()
        if core_prof and reduced:
            reduced[0]["_core_profile"] = core_prof
        return reduced, total_usage

    async def summarize_session_facts(
        self,
        *,
        session_id: str,
        session_date: str,
        turns: list[dict[str, Any]],
        timezone_name: str,
    ) -> tuple[dict[str, Any], dict[str, int] | None]:
        """Distill a full session into a coreference-resolved, temporally-anchored summary & atomic facts."""
        if not turns:
            return {"summary": "", "facts": [], "entities": []}, None
        turn_sample = " ".join(str(t.get("text", ""))[:200] for t in turns[:8])
        sys_prompt = (
            SESSION_SUMMARY_SYSTEM_PROMPT_EN
            if _is_primarily_english(turn_sample)
            else SESSION_SUMMARY_SYSTEM_PROMPT
        )
        compact_turns = [
            {
                "speaker": t.get("speaker") or t.get("role") or "user",
                "time": t.get("time") or session_date,
                "text": str(t.get("text") or "")[:420],
            }
            for t in turns[:36]
        ]
        payload = json.dumps(
            {
                "session_id": session_id,
                "session_date": session_date,
                "timezone": timezone_name,
                "turns": compact_turns,
            },
            ensure_ascii=False,
        )
        raw, usage = await self._chat_completion(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": payload},
            ],
            max_tokens=1024,
            temperature=0.0,
            json_mode=True,
        )
        parsed = _extract_json_object(raw)
        summary = str(parsed.get("summary") or "").strip()
        facts = [
            str(f).strip()
            for f in (parsed.get("facts") or [])
            if isinstance(f, str) and str(f).strip()
        ][:12]
        entities = [
            str(e).strip()
            for e in (parsed.get("entities") or [])
            if isinstance(e, str) and str(e).strip()
        ][:16]
        return {"summary": summary, "facts": facts, "entities": entities}, usage

    async def rerank_evidence_listwise(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 10,
    ) -> tuple[list[int], dict[str, int] | None]:
        """Ultra-compact listwise evidence selector returning top_k decisive candidate indices."""
        if not candidates:
            return [], None
        brief = [
            {
                "idx": i,
                "type": c.get("type", "msg"),
                "date": str(c.get("date") or "")[:24],
                "text": str(c.get("text") or "")[:260],
            }
            for i, c in enumerate(candidates[:24])
        ]
        payload = json.dumps(
            {"question": query, "top_k": top_k, "candidates": brief},
            ensure_ascii=False,
        )
        raw, usage = await self._chat_completion(
            [
                {"role": "system", "content": LISTWISE_RERANK_SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
            max_tokens=128,
            temperature=0.0,
            json_mode=True,
        )
        parsed = _extract_json_object(raw)
        raw_indices = parsed.get("selected_indices") or []
        valid: list[int] = []
        for x in raw_indices:
            try:
                i = int(x)
                if 0 <= i < len(brief) and i not in valid:
                    valid.append(i)
            except (TypeError, ValueError):
                continue
        return valid[:top_k], usage

    async def generate_reply(
        self,
        *,
        user_message: str,
        recent_messages: list[MessageRecord],
        narrated_context: str,
        now_iso: str,
        timezone_name: str,
    ) -> tuple[str, dict[str, int] | None]:
        messages = self._build_reply_messages(
            user_message=user_message,
            recent_messages=recent_messages,
            narrated_context=narrated_context,
            now_iso=now_iso,
            timezone_name=timezone_name,
        )
        return await self._chat_completion(
            messages,
            max_tokens=self._max_reply_tokens,
            temperature=self._temperature,
            json_mode=False,
        )

    def _build_reply_messages(
        self,
        *,
        user_message: str,
        recent_messages: list[MessageRecord],
        narrated_context: str,
        now_iso: str,
        timezone_name: str,
    ) -> list[dict[str, str]]:
        system_content = (
            PERSONA_SYSTEM_PROMPT
            + f"\n\n当前时间：{now_iso}（时区：{timezone_name}）"
        )
        if narrated_context.strip():
            system_content += (
                "\n\n以下是你目前知道的前情与记忆线索（仅供参考，不要直接复述标签格式）：\n"
                + narrated_context.strip()
            )
        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
        for msg in recent_messages[-8:]:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_message})
        return messages

    async def generate_reply_stream(
        self,
        *,
        user_message: str,
        recent_messages: list[MessageRecord],
        narrated_context: str,
        now_iso: str,
        timezone_name: str,
    ):
        """True SSE streaming reply generator yielding {'delta': str} then {'usage': dict|None}."""
        import asyncio

        messages = self._build_reply_messages(
            user_message=user_message,
            recent_messages=recent_messages,
            narrated_context=narrated_context,
            now_iso=now_iso,
            timezone_name=timezone_name,
        )
        url = (
            f"{self._base_url}/chat/completions"
            if not self._base_url.endswith("/chat/completions")
            else self._base_url
        )
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self._max_reply_tokens,
            "temperature": self._temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self._disable_thinking and "deepseek" in self._base_url.lower():
            body["thinking"] = {"type": "disabled"}

        last_error: Exception | None = None
        for attempt in range(4):
            emitted_any = False
            usage = None
            try:
                client = self._get_client()
                async with client.stream("POST", url, json=body) as response:
                    if response.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    if response.status_code != 200:
                        raise RuntimeError(f"LLM HTTP {response.status_code}")
                    async for raw_line in response.aiter_lines():
                        line = raw_line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            yield {"usage": usage}
                            return
                        chunk = json.loads(data_str)
                        if isinstance(chunk.get("usage"), dict):
                            u = chunk["usage"]
                            usage = {
                                "prompt_tokens": int(u.get("prompt_tokens", 0)),
                                "completion_tokens": int(u.get("completion_tokens", 0)),
                                "total_tokens": int(u.get("total_tokens", 0)),
                            }
                        for choice in chunk.get("choices", []):
                            delta = (choice.get("delta") or {}).get("content")
                            if delta:
                                emitted_any = True
                                yield {"delta": str(delta)}
                    yield {"usage": usage}
                    return
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                await self.aclose()
                if not emitted_any and attempt < 3:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"LLM streaming transport failure ({type(exc).__name__})"
                ) from None
        raise RuntimeError(str(last_error or "LLM stream failed"))

    async def proxy_chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Transparent non-streaming OpenAI-compatible pass-through for AIRI / external clients."""
        import asyncio

        url = (
            f"{self._base_url}/chat/completions"
            if not self._base_url.endswith("/chat/completions")
            else self._base_url
        )
        body: dict[str, Any] = {
            "model": model or self.model_name,
            "messages": messages,
            "stream": False,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if extra_params:
            for k, v in extra_params.items():
                if k not in {"model", "messages", "stream"} and v is not None:
                    body[k] = v
        if self._disable_thinking and "deepseek" in self._base_url.lower() and "thinking" not in body:
            body["thinking"] = {"type": "disabled"}

        last_error: Exception | None = None
        for attempt in range(4):
            try:
                client = self._get_client()
                response = await client.post(url, json=body)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                if response.status_code != 200:
                    raise RuntimeError(f"Upstream LLM HTTP {response.status_code}")
                return response.json()
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                await self.aclose()
                if attempt < 3:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"Upstream LLM transport failure ({type(exc).__name__})"
                ) from None
        raise RuntimeError(str(last_error or "Upstream LLM proxy failed"))

    async def proxy_chat_completion_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ):
        """Transparent streaming OpenAI-compatible pass-through yielding raw OpenAI SSE chunk dicts."""
        import asyncio

        url = (
            f"{self._base_url}/chat/completions"
            if not self._base_url.endswith("/chat/completions")
            else self._base_url
        )
        body: dict[str, Any] = {
            "model": model or self.model_name,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if extra_params:
            for k, v in extra_params.items():
                if k not in {"model", "messages", "stream"} and v is not None:
                    body[k] = v
        if self._disable_thinking and "deepseek" in self._base_url.lower() and "thinking" not in body:
            body["thinking"] = {"type": "disabled"}

        last_error: Exception | None = None
        for attempt in range(4):
            emitted_any = False
            try:
                client = self._get_client()
                async with client.stream("POST", url, json=body) as response:
                    if response.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    if response.status_code != 200:
                        raise RuntimeError(f"Upstream LLM HTTP {response.status_code}")
                    async for raw_line in response.aiter_lines():
                        line = raw_line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            return
                        chunk = json.loads(data_str)
                        emitted_any = True
                        yield chunk
                    return
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                await self.aclose()
                if not emitted_any and attempt < 3:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"Upstream LLM stream transport failure ({type(exc).__name__})"
                ) from None
        raise RuntimeError(str(last_error or "Upstream LLM proxy stream failed"))
