"use strict";
const $ = id => document.getElementById(id);
const state = {csrf:null, chat:null, chats:[], messages:[], active:null, source:null, prefs:null,
  chatCursor:null, messageCursor:null, memoryCursor:null, pendingSend:null, forget:null};
const terminal = ["turn.completed", "turn.cancelled", "turn.failed", "turn.invalidated"];
const errors = {companion_control_capacity:"互动边界记录已满，请先在记忆管理中忘记不再需要的设置。",companion_output_blocked:"回复未通过当前互动边界检查，请重新发送。",unauthorized:"登录已过期，请重新登录。",invalid_origin:"请从本地服务地址打开页面。",invalid_csrf:"页面状态已过期，请刷新后重试。",login_rate_limited:"登录尝试较多，请稍后重试。",turn_in_progress:"还有一个回复正在进行，可以先停止它。",preference_revision_conflict:"偏好已在其他页面更新，请核对后重新保存。",stale_lease:"回复已失效，请刷新历史。",memory_revision_conflict:"记忆已更新，请重新查看来源。",source_revision_conflict:"原话已更新，请重新预览。",source_deleted:"这条内容已忘记，请刷新历史。",invalid_request:"输入格式不正确，请检查后重试。",database_migration_required:"数据库需要升级，请先运行 hmph migrate。"};
function notify(text){$("notification").textContent=text;$("notification").hidden=false;clearTimeout(notify.timer);notify.timer=setTimeout(()=>$("notification").hidden=true,6500);}
function closeStream(){if(state.source)state.source.close();state.source=null;}
function signedOut(){closeStream();state.csrf=null;state.active=null;state.messages=[];$("app").hidden=true;$("login").hidden=false;$("token").value="";$("messages").replaceChildren();$("memories").replaceChildren();$("shared-threads").replaceChildren();$("short-memory").replaceChildren();$("memory-health").textContent="";$("working-memory").replaceChildren();$("thread-topic").value="";}
async function api(path,options={}){
  const headers={...options.headers};if(options.body){headers["Content-Type"]="application/json";if(state.csrf)headers["X-CSRF-Token"]=state.csrf;}
  const response=await fetch(path,{credentials:"same-origin",cache:"no-store",...options,headers});
  const data=await response.json();if(!response.ok){if(response.status===401)signedOut();const code=data.error?.code||"request_failed";throw new Error(errors[code]||`操作未完成：${code}`);}return data;
}
const write=(method,path,body)=>api(path,{method,body:JSON.stringify(body)});
function task(action){return async event=>{event?.preventDefault();try{await action(event);}catch(error){notify(error.message||"网络连接失败，请重试。");}};}
function busy(){const active=!!state.active;$("send").disabled=active||!state.chat;$("stop").hidden=!active;$("older").disabled=active;$("busy-banner").hidden=!active||state.active.conversation_id===state.chat;}
function renderChats(){const root=$("chats");root.replaceChildren();for(const [index,chat]of state.chats.entries()){const button=document.createElement("button");button.className="chat-link"+(chat.id===state.chat?" selected":"");button.textContent=new Date(chat.created_at).toLocaleString("zh-CN",{month:"short",day:"numeric",hour:"2-digit",minute:"2-digit"});button.title=`对话 ${index+1}`;button.onclick=task(()=>selectChat(chat.id));root.append(button);}$("more-chats").hidden=!state.chatCursor;}
async function loadChats(append=false){const query=new URLSearchParams({limit:"30"});if(append&&state.chatCursor)query.set("cursor",state.chatCursor);const page=await api(`/api/conversations/page?${query}`);state.chats=append?[...state.chats,...page.items]:page.items;state.chatCursor=page.next_cursor;renderChats();}
function bubble(role,text,deleted=false){const row=document.createElement("div");row.className=`message-row ${role}`;const name=document.createElement("span");name.className="speaker";name.textContent=role==="user"?"你":"HMPH";const content=document.createElement("div");content.className="bubble"+(deleted?" deleted":"");content.textContent=text;row.append(name,content);$("messages").append(row);return content;}
function renderMessages(){const root=$("messages");root.replaceChildren();for(const message of state.messages){const node=bubble(message.role,message.content,message.deleted);if(message.role==="user"&&!message.deleted){const forget=document.createElement("button");forget.textContent="忘记这条";forget.className="message-forget";forget.onclick=task(()=>previewForget(`/api/messages/${message.id}`));node.parentElement.append(forget);}}if(!state.messages.length){const empty=document.createElement("p");empty.className="empty";empty.textContent="不用准备开场白。想到什么，就从那里开始。";root.append(empty);}$("older").hidden=!state.messageCursor;}
async function loadMessages(older=false){if(!state.chat)return;const query=new URLSearchParams({limit:"50"});if(older&&state.messageCursor)query.set("cursor",state.messageCursor);const page=await api(`/api/conversations/${state.chat}/messages/page?${query}`);state.messages=older?[...page.items,...state.messages]:page.items;state.messageCursor=page.next_cursor;renderMessages();if(!older)$("scroll").scrollTop=$("scroll").scrollHeight;}
async function selectChat(id){closeStream();state.chat=id;await refreshMode();history.replaceState(null,"",`#chat=${id}`);renderChats();await loadMessages();busy();if(state.active?.conversation_id===id)connect(state.active);}
async function createChat(){const chat=await write("POST","/api/conversations",{request_id:crypto.randomUUID()});await loadChats();await selectChat(chat.id);}
async function refreshActive(){state.active=await api("/api/active-turn");busy();}
async function finishStream(){closeStream();state.active=null;$("connection").textContent="就绪";busy();await refreshMode();await loadMessages();}
function connect(turn){closeStream();state.active=turn;busy();if(turn.conversation_id!==state.chat)return;for(const node of [...$("messages").children])if(node.classList.contains("empty"))node.remove();
  // Replay from zero replaces the active draft, so reloading never duplicates a prefix.
  const existing=state.messages.find(m=>m.turn_id===turn.id&&m.role==="assistant");if(existing){const index=state.messages.indexOf(existing);$("messages").children[index]?.remove();}
  let output=null;const seen=new Set();const source=new EventSource(`/api/turns/${turn.id}/events`);state.source=source;$("connection").textContent="正在等待回复";
  source.addEventListener("text.delta",event=>{if(seen.has(event.lastEventId))return;seen.add(event.lastEventId);const data=JSON.parse(event.data);if(!output)output=bubble("assistant","");output.textContent+=data.payload.text;$("connection").textContent="正在回复";$("scroll").scrollTop=$("scroll").scrollHeight;});
  for(const type of terminal)source.addEventListener(type,task(async event=>{const data=JSON.parse(event.data);if(type!=="turn.completed")notify(type==="turn.cancelled"?"回复已停止。":`回复已结束：${data.payload.error_code||type}`);await finishStream();}));
  source.addEventListener("turn.content_redacted",task(async()=>{notify("相关内容已忘记，正在刷新历史。");await finishStream();}));
  source.addEventListener("session.expired",()=>{signedOut();notify("登录已过期，请重新登录。");});
  source.onopen=()=>$("connection").textContent="连接已恢复";
  source.onerror=()=>{if(state.source===source)$("connection").textContent="连接中断，正在重连";};
}
async function refreshMode(){const choice=await api("/api/memory-selection");$("mode").textContent=`${state.modelLabel} · ${choice.mode==="lexical"?"词法记忆基线":choice.available?"增强记忆实验（异常时回退）":"增强记忆暂不可用，使用原文回退"}`;}
async function start(){state.chat=null;busy();const session=await api("/api/session");state.csrf=session.csrf_token;$("login").hidden=true;$("app").hidden=false;const health=await api("/health");state.modelLabel=health.model_mode==="fake"?"测试模型":"DeepSeek";await loadChats();await refreshActive();const hash=new URLSearchParams(location.hash.slice(1));let chosen=state.active?.conversation_id||hash.get("chat")||state.chats[0]?.id;if(chosen){try{await selectChat(chosen);}catch{if(state.chats.length)await selectChat(state.chats[0].id);else await createChat();}}else await createChat();}
$("login-form").onsubmit=task(async()=>{const token=$("token").value;$("token").value="";await write("POST","/api/session",{token});await start();});
$("logout").onclick=task(async()=>{await write("POST","/api/session/logout",{});signedOut();});
$("new-chat").onclick=task(createChat);$("more-chats").onclick=task(()=>loadChats(true));$("older").onclick=task(()=>loadMessages(true));$("jump-active").onclick=task(()=>selectChat(state.active.conversation_id));
$("composer").onsubmit=task(async()=>{const content=$("message").value;if(!content.trim()||state.active)return;const key=state.pendingSend;if(!key||key.chat!==state.chat||key.content!==content)state.pendingSend={chat:state.chat,content,request_id:crypto.randomUUID()};try{const turn=await write("POST",`/api/conversations/${state.chat}/turns`,{request_id:state.pendingSend.request_id,content});state.pendingSend=null;$("message").value="";await loadMessages();connect(turn);}catch(error){await refreshActive();throw error;}});
$("message").onkeydown=event=>{if(event.key==="Enter"&&!event.shiftKey&&!event.isComposing){event.preventDefault();$("composer").requestSubmit();}};
$("stop").onclick=task(async()=>{if(state.active)await write("POST",`/api/turns/${state.active.id}/cancel`,{request_id:crypto.randomUUID()});await finishStream();});
function openPanel(kind){$("panel").hidden=false;$("memory-panel").hidden=kind!=="memory";$("preferences-panel").hidden=kind!=="preferences";$("panel-title").textContent=kind==="memory"?"记忆与来源":"表达偏好";}
$("close-panel").onclick=()=>$("panel").hidden=true;
async function loadPreferences(){state.prefs=await api("/api/preferences");const p=state.prefs.preferences;$("nickname").value=p.nickname||"";$("reply-length").value=p.reply_length;$("playfulness").value=p.playfulness;$("support-mode").value=p.support_mode;}
$("preferences-toggle").onclick=task(async()=>{openPanel("preferences");await loadPreferences();});
$("preferences-form").onsubmit=task(async()=>{try{state.prefs=await write("PUT","/api/preferences",{request_id:crypto.randomUUID(),conversation_id:state.chat,expected_revision:state.prefs.revision,preferences:{nickname:$("nickname").value.trim()||null,reply_length:$("reply-length").value,playfulness:$("playfulness").value,support_mode:$("support-mode").value}});notify("偏好已保存；进行中的旧回复会失效。");await refreshActive();await loadMessages();}catch(error){await loadPreferences();throw error;}});
async function loadHistory(memory,root,cursor=null){
  if(cursor===null)root.replaceChildren();const query=new URLSearchParams({limit:"10"});if(cursor)query.set("cursor",cursor);
  const page=await api(`/api/memories/${memory.id}/history/page?${query}`);
  for(const version of page.items){const block=document.createElement("section");const title=document.createElement("p");title.textContent=`${version.scope==="historical"?"历史版本":"记录"} · ${new Date(version.recorded_at).toLocaleString()} · ${version.value}`;const quotes=document.createElement("p");quotes.textContent=version.sources.map(s=>s.quote).join("\n");block.append(title,quotes);
    if(version.sources.length<version.total_sources){const more=document.createElement("button");more.textContent=`更多来源（共 ${version.total_sources} 条）`;let next=version.sources.at(-1)?.message_id;more.onclick=task(async()=>{more.disabled=true;try{const params=new URLSearchParams({revision:String(version.revision),limit:"20"});if(next)params.set("cursor",next);const sources=await api(`/api/memories/${memory.id}/sources?${params}`);quotes.textContent+="\n"+sources.items.map(s=>s.quote).join("\n");next=sources.next_cursor;if(!next)more.remove();}finally{more.disabled=false;}});block.append(more);}root.append(block);}
  if(page.next_cursor){const more=document.createElement("button");more.textContent="更早的版本";more.onclick=task(async()=>{more.disabled=true;try{await loadHistory(memory,root,page.next_cursor);more.remove();}finally{more.disabled=false;}});root.append(more);}
}
async function previewForget(base){const preview=await api(`${base}/forget-preview`);state.forget={base,preview,request_id:crypto.randomUUID()};$("forget-scope").textContent=`预计影响 ${preview.memories} 条记忆、${preview.messages} 条消息、${preview.turns} 个回合。${preview.framework_spaces?"包含框架派生数据，清理可能需要等待。":""}`;$("forget-dialog").showModal();}
async function loadMemories(append=false){if(!append){await loadSharedThreads();await loadMemoryState();await loadMemoryHealth();}const query=new URLSearchParams({limit:"30"});if(append&&state.memoryCursor)query.set("cursor",state.memoryCursor);const page=await api(`/api/memories/page?${query}`);state.memoryCursor=page.next_cursor;if(!append)$("memories").replaceChildren();for(const memory of page.items){const card=document.createElement("article");card.className="memory-card";const label=document.createElement("span");label.className="memory-label";label.textContent=({expression_preferences:"表达偏好记录",companion_boundary:"互动边界记录",companion_repair:"互动修复记录"})[memory.predicate]||(memory.kind==="episode"?"经历":"你说过的事实");const value=document.createElement("p");value.className="memory-value";value.textContent=memory.value;const sources=document.createElement("button");sources.textContent="查看来源";const details=document.createElement("div");details.className="sources";details.hidden=true;sources.onclick=task(async()=>{if(details.hidden){await loadHistory(memory,details);}details.hidden=!details.hidden;});const forget=document.createElement("button");forget.textContent="忘记";forget.onclick=task(()=>previewForget(`/api/memories/${memory.id}`));card.append(label,value,sources,forget,details);$("memories").append(card);}if(!append&&!page.items.length){const empty=document.createElement("p");empty.className="muted";empty.textContent="暂时没有长期记忆。测试模型不会自动整理聊天记忆；真实模式整理完成后会显示在这里。";$("memories").append(empty);}$("more-memories").hidden=!state.memoryCursor;}
$("memory-toggle").onclick=task(async()=>{openPanel("memory");await loadMemories();});$("refresh-memories").onclick=task(()=>loadMemories());$("more-memories").onclick=task(()=>loadMemories(true));$("forget-cancel").onclick=()=>$("forget-dialog").close();
async function watchDeletion(receipt,node){if(!state.csrf)return;node.textContent=receipt.status==="completed"?"已完成本地清理。":"相关记忆已停止使用，派生数据仍在清理中。";if(receipt.status!=="completed")setTimeout(async()=>{try{await watchDeletion(await api(`/api/memory-deletions/${receipt.id}`),node);}catch(error){notify(error.message);}},2000);}
$("forget-confirm").onclick=task(async()=>{const selected=state.forget;if(!selected)return;$("forget-confirm").disabled=true;try{const receipt=await write("POST",`${selected.base}/forget`,{request_id:selected.request_id,expected_revision:selected.preview.revision});$("forget-dialog").close();state.forget=null;const node=document.createElement("p");node.className="notice";$("deletions").append(node);watchDeletion(receipt,node);await refreshActive();await loadMemories();await loadMessages();}finally{$("forget-confirm").disabled=false;}});
start().catch(error=>{signedOut();if(!error.message.includes("登录已过期"))notify(error.message);});

async function loadSharedThreads(){
  const page=await api("/api/shared-threads");
  const root=$("shared-threads");root.replaceChildren();
  const labels={open:"进行中",paused:"已暂停",resolved:"已完成",dropped:"已放下"};
  for(const thread of page.items){
    const card=document.createElement("article");card.className="memory-card";
    const title=document.createElement("p");title.textContent=`${thread.topic} · ${labels[thread.status]}`;
    card.append(title);
    const details=document.createElement("div");details.hidden=true;
    const sources=document.createElement("button");sources.textContent="查看原话";
    sources.onclick=task(async()=>{if(details.hidden){const page=await api(`/api/shared-threads/${thread.id}/sources`);details.replaceChildren();for(const source of page.items){const line=document.createElement("p");line.textContent=`${new Date(source.recorded_at).toLocaleString()} · ${source.content}`;details.append(line);}}details.hidden=!details.hidden;});
    card.append(sources,details);
    const choices=thread.status==="open"?[["paused","暂停"],["resolved","完成"],["dropped","放下"]]:thread.status==="paused"?[["open","恢复"],["resolved","完成"],["dropped","放下"]]:[];
    for(const [status,label] of choices){
      const button=document.createElement("button");button.textContent=label;
      button.onclick=task(async()=>{button.disabled=true;try{
        await write("POST",`/api/shared-threads/${thread.id}`,{conversation_id:state.chat,request_id:crypto.randomUUID(),expected_revision:thread.revision,status});
        await refreshActive();await loadSharedThreads();await loadMessages();
      }finally{button.disabled=false;}});card.append(button);
    }
    const forget=document.createElement("button");forget.textContent="查看忘记范围";
    forget.onclick=task(()=>previewForget(`/api/messages/${thread.source_message_id}`));
    card.append(forget);root.append(card);
  }
}
async function loadMemoryHealth(){
  const root=$("memory-health");
  try{
    const health=await api("/api/memory-health");
    const mode=health.mode==="experimental_index"?`已选择实验索引，其快照后有 ${health.new_source_count} 条新原话（并非都需要保存）。`:health.mode==="fallback"?"选定的实验索引暂不可用，回复使用原话检索。":"目前使用原话检索，未启用增强索引。";
    const pending=health.pending_extractions?`还有 ${health.pending_extractions} 项原话等待整理。`:"";
    const failed=health.failed_extractions?`有 ${health.failed_extractions} 项整理失败，可在来源任务中查看。`:"";
    const building=({queued:"候选记忆正在排队整理，期间仍使用当前可用来源。",building:"候选记忆正在整理，期间仍使用当前可用来源。",ready:"候选记忆已整理完成，不会自动改变聊天选择。",failed:"本次候选整理失败，现有可用来源仍保留。",retired:"旧候选已退役，回复使用当前可用来源。"})[health.latest_build_status]||"";
    const maintenance=health.maintenance;
    const upkeep=maintenance?.enabled?`后台整理已明确配置，剩余调用预留额度 ${maintenance.remaining_calls}。仅在维护进程运行时处理，结果不自动用于聊天。`:maintenance?.suspended_reason?"后台整理已暂停，请查看构建状态；不会自动追加调用额度。":"后台整理未启用，不会自动付费重建索引。";
    root.textContent=mode+pending+failed+building+upkeep;
  }catch{root.textContent="暂时无法读取整理进度；仍可查看或忘记记忆。";}
}
async function loadMemoryState(){
  const root=$("short-memory");root.replaceChildren();
  if(!state.chat)return;
  const snapshot=await api(`/api/conversations/${state.chat}/state`);
  const hint=document.createElement("p");hint.className="muted";
  hint.textContent=snapshot.pending_messages?"还有新的原话尚未整理；暂存状态可能落后。":"以下是本会话的暂存线索，可能理解有误。";root.append(hint);
  const atoms=[["当前话题",snapshot.data.topic],["当前目标",snapshot.data.goal],...snapshot.data.open_items.map(item=>[({open:"待继续",resolved:"已解决",dropped:"已放下"})[item.status],item.content])];
  for(const [label,atom]of atoms){if(!atom)continue;const line=document.createElement("p");line.textContent=`${label}：${atom.text}`;const quote=document.createElement("blockquote");quote.textContent=atom.quote;root.append(line,quote);if(state.messages.some(m=>m.id===atom.message_id&&m.role==="user"&&!m.deleted)){const forget=document.createElement("button");forget.textContent="忘记这条依据";forget.onclick=task(()=>previewForget(`/api/messages/${atom.message_id}`));root.append(forget);}}
  if(!atoms.some(([,atom])=>atom)){const empty=document.createElement("p");empty.textContent="暂时没有已整理的短期线索。";root.append(empty);}
  const target=$("working-memory");target.replaceChildren();
  const last=[...state.messages].reverse().find(m=>m.role==="assistant"&&!m.deleted);
  if(!last){target.textContent="聊天回复后，这里会显示本轮记忆线索。";return;}
  const explanation=await api(`/api/turns/${last.turn_id}/memory-context`);
  const workspace=explanation.context?.workspace;
  if(!workspace){target.textContent="这个旧回合没有保存工作记忆摘要。";return;}
  const line=document.createElement("p");line.textContent=`本轮选入 ${workspace.evidence_count} 项记忆线索。${workspace.pending_gap?"仍有未覆盖的新原话。":workspace.coverage==="complete"?"本轮目标有可用线索；不表示全部历史都已找齐。":"可能有遗漏，找不到不代表你从未说过。"}`;target.append(line);
  if(workspace.state_hint_count){const cue=document.createElement("p");cue.textContent="这轮检索借助了有原话依据的短期目标来理解指代。";target.append(cue);}
}
$("thread-form").onsubmit=task(async()=>{
  const topic=$("thread-topic").value.trim();if(!topic||!state.chat)return;
  const button=$("thread-form").querySelector("button");button.disabled=true;
  try{
    await write("POST","/api/shared-threads",{conversation_id:state.chat,request_id:crypto.randomUUID(),topic});
    $("thread-topic").value="";await refreshActive();await loadSharedThreads();await loadMessages();
  }finally{button.disabled=false;}
});
