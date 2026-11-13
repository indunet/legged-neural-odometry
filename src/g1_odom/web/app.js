const $=id=>document.getElementById(id);let paused=false,lastState=null;
const isTraining=location.pathname==='/training';
$('simulation').hidden=isTraining;$('training').hidden=!isTraining;
document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('active',a.pathname===(location.pathname==='/'?'/odometry':location.pathname)));
function message(text){$('message').textContent=text;$('message').style.display='block';setTimeout(()=>$('message').style.display='none',4500)}
async function post(path,body={}){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const j=await r.json();if(!r.ok)throw Error(j.error||r.status);return j}
function action(id,fn){$(id).onclick=()=>Promise.resolve().then(fn).catch(e=>message(e.message))}
action('start',()=>post('/api/sim/start'));action('stop',()=>post('/api/sim/stop'));
action('pause',()=>{paused=!paused;return post('/api/sim/control',{paused})});action('reset',()=>post('/api/sim/control',{reset:true}));
action('zero',()=>{['vx','vy','wz'].forEach(id=>{$(id).value=0;$(id).nextElementSibling.textContent='0.00'});return post('/api/sim/control',{command:[0,0,0]})});
action('resume',async()=>{const r=await post('/api/training/start');message(r.message||'流程已启动')});action('stopTrain',async()=>{await post('/api/training/stop');message('流程已停止，可恢复')});
for(const id of ['vx','vy','wz']){$(id).oninput=()=>{$(id).nextElementSibling.textContent=Number($(id).value).toFixed(2)};$(id).onchange=()=>post('/api/sim/control',{command:['vx','vy','wz'].map(x=>Number($(x).value))}).catch(e=>message(e.message))}
function plot(canvas,series,xy=true,circles=[]){
 const cw=canvas.clientWidth,ch=canvas.clientHeight;if(!cw||!ch)return;
 const ratio=Math.min(window.devicePixelRatio||1,2);
 if(canvas.width!==Math.round(cw*ratio)||canvas.height!==Math.round(ch*ratio)){canvas.width=Math.round(cw*ratio);canvas.height=Math.round(ch*ratio)}
 const ctx=canvas.getContext('2d'),w=cw,h=ch,pad=28;ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);
 const points=series.flatMap(s=>s.points);for(const c of circles)points.push([c.x-c.r,c.y-c.r],[c.x+c.r,c.y+c.r]);if(!points.length)return;
 const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
 let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
 let dx=Math.max(xmax-xmin,xy?.5:.02),dy=Math.max(ymax-ymin,xy?.5:.02);
 if(xy){const unit=Math.max(dx/(w-2*pad),dy/(h-2*pad));dx=unit*(w-2*pad);dy=unit*(h-2*pad)}
 const cx=(xmin+xmax)/2,cy=(ymin+ymax)/2;xmin=cx-dx*.55;ymin=cy-dy*.55;dx*=1.1;dy*=1.1;
 const X=x=>pad+(x-xmin)/dx*(w-2*pad),Y=y=>h-pad-(y-ymin)/dy*(h-2*pad);
 ctx.strokeStyle='#ffffff1f';ctx.lineWidth=1;
 for(let i=0;i<=4;i++){const x=pad+i*(w-2*pad)/4,y=pad+i*(h-2*pad)/4;ctx.beginPath();ctx.moveTo(x,pad);ctx.lineTo(x,h-pad);ctx.stroke();ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(w-pad,y);ctx.stroke()}
 ctx.lineJoin='round';
 for(const s of series){ctx.strokeStyle=s.color;ctx.lineWidth=2;ctx.beginPath();s.points.forEach((q,i)=>{i?ctx.lineTo(X(q[0]),Y(q[1])):ctx.moveTo(X(q[0]),Y(q[1]))});ctx.stroke()}
 for(const c of circles){ctx.strokeStyle=c.color;ctx.fillStyle=c.color+'33';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(X(c.x),Y(c.y),c.r/dx*(w-2*pad),0,2*Math.PI);ctx.fill();ctx.stroke()}
 ctx.fillStyle='#9fb0c6';ctx.font='11px system-ui';ctx.fillText(`${xmin.toFixed(xy?1:2)} → ${(xmin+dx).toFixed(xy?1:2)} ${xy?'m':'s'}`,pad,h-8);
 if(!xy)ctx.fillText(`${ymin.toFixed(3)} → ${(ymin+dy).toFixed(3)}`,pad,14)}
const QUALITY={ok:'正常',warn:'注意',alarm:'告警'};
const cm=v=>v==null?'—':(100*v).toFixed(2),pct=v=>v==null?'—':(100*v).toFixed(1)+'%';
function render(s){$('headingReference').textContent=s.heading_reference||'航向：IMU 提供 yaw；仿真使用理想姿态。';lastState=s;$('status').textContent=({idle:'未启动',starting:'启动中',stopped:'已结束',running:'运行中',paused:'已暂停',fallen:'已摔倒',error:'运行异常'})[s.status]||s.status||'就绪';if(s.error)$('status').textContent=s.error;const rows=s.samples||[];if(rows.length){const r=rows.at(-1),dx=r.pose[0]-r.truth[0],dy=r.pose[1]-r.truth[1];$('xy').textContent=`${(Math.hypot(dx,dy)*100).toFixed(1)} cm`;$('sigma').textContent=r.sigma_xy==null?'—':`${(r.sigma_xy*100).toFixed(1)} cm`;$('quality').textContent=QUALITY[r.quality]||'—';$('quality').className=r.quality||'';$('time').textContent=`${r.t.toFixed(1)} s`;const track=(s.track||rows.map(r=>[r.truth[0],r.truth[1],r.pose[0],r.pose[1]])).concat([[r.truth[0],r.truth[1],r.pose[0],r.pose[1]]]);plot($('trajectory'),[{color:'#b6c2d3',points:track.map(p=>[p[0],p[1]])},{color:'#5aa2ff',points:track.map(p=>[p[2],p[3]])}],true,r.sigma_xy==null?[]:[{x:r.pose[0],y:r.pose[1],r:2*r.sigma_xy,color:'#ffb84d'}]);const tail=rows.slice(-750);plot($('curves'),[{color:'#5aa2ff',points:tail.map(r=>[r.t,r.velocity[0]])},{color:'#43d3b0',points:tail.map(r=>[r.t,r.velocity[1]])}],false);plot($('errorCurve'),[{color:'#ff7b86',points:tail.map(r=>[r.t,Math.hypot(r.pose[0]-r.truth[0],r.pose[1]-r.truth[1])])},{color:'#ffb84d',points:tail.filter(r=>r.sigma_xy!=null).map(r=>[r.t,2*r.sigma_xy])}],false)}
 const d=s.data||{};$('dataCount').textContent=d.total?`${d.done} / ${d.total}`:'等待数据';$('dataBar').max=d.total||1;$('dataBar').value=d.done||0;
 const p=s.pipeline||{},stage={collecting:'数据准备中',auditing:'数据审计',label_audit:'标签审计',training_mean:'训练均值网络',training_variance:'训练方差头',trained:'训练完成，等待校准',calibrating:'不确定度校准',calibrated:'校准完成，等待测试',evaluating:'独立测试',complete:'训练与验收完成',stopped:'已停止'};
 $('trainStage').textContent=stage[p.status]||p.status||'等待流程';$('epoch').textContent=p.epoch?`窗口 ${p.window} · 种子 ${p.seed} · epoch ${p.epoch} / ${p.epochs}`:'';$('record').textContent=p.record?JSON.stringify(p.record,null,2):'当前没有正在训练的网络';$('models').textContent=s.released?'已发布':'未发布'}
let session=null;
async function poll(){try{const r=await fetch('/api/state');const s=await r.json();render(s);
 if(!isTraining&&s.session&&s.session!==session&&['starting','running','paused','fallen'].includes(s.status)){session=s.session;$('frame').src=`/api/stream?s=${session}`}}
 catch(e){$('status').textContent='连接已断开'}setTimeout(poll,500)}
$('frame').onload=()=>{$('frame').style.display='block';$('empty').style.display='none'};
function table(head,rows){const t=document.createElement('table');t.innerHTML='<thead><tr>'+head.map(h=>`<th>${h}</th>`).join('')+'</tr></thead>';for(const values of rows){const row=t.insertRow();for(const v of values)row.insertCell().textContent=v}return t}
function element(tag,text){const e=document.createElement(tag);e.textContent=text;return e}
async function reports(){
 if(!isTraining)return;
 try{
  const j=await (await fetch('/api/reports')).json();
  if(j.datasets){
   const sets=Object.entries(j.datasets),c=j.calibration||{},det=j.detection||{};
   $('modelHash').textContent=`SHA256 ${j.sha256.slice(0,12)} · 三种子集成，输出速度与方差`;
   const contents=[element('p','航向使用理想仿真 IMU yaw，不代表宇树真机精度；终点与 ATE 对照真实骨盆位置，只做初始对齐。'),
    element('h3','精度'),
    table(['测试集','轨迹／跌倒','终点均值 (cm)','终点 P95 (cm)','ATE (cm)','5 m RPE (cm)','不加滞后补偿 (cm)'],
     sets.map(([n,o])=>[n,`${o.episodes} / ${o.fell}`,cm(o.endpoint_xy_m.mean),cm(o.endpoint_xy_m.p95),cm(o.ate_xy_m.mean),cm(o.rpe_5m),cm(o.no_lag_endpoint_mean_m)])),
    element('h3','不确定度'),
    element('p',`校准（只用验证集）：偶然 ×${c.aleatoric_scale?.toFixed(3)}，种子间 ×${c.epistemic_scale?.toFixed(3)}，相关时间 ${c.correlation_time_s?.toFixed(3)} s，偏置占比 ${c.bias_fraction?.toFixed(3)}，位姿缩放 ${c.pose_scale?.toFixed(3)}。`),
    table(['测试集','速度 1σ 覆盖','速度 2σ 覆盖','位姿 2σ 覆盖','位姿 NEES 均值','warn+alarm 帧'],
     sets.map(([n,o])=>[n,pct(o.coverage_1sigma),pct(o.coverage_2sigma),pct(o.pose_coverage_2sigma),o.pose_nees_mean?.toFixed(2)??'—',pct(o.warn_or_alarm)])),
    element('p',`异常检出 AUROC：分布外 vs 分布内 ${det.auroc_ood_vs_iid?.toFixed(3)}；高误差轨迹 ${det.auroc_high_error_episodes?.toFixed(3)}。理想 NEES 均值为 2，位姿 2σ 覆盖目标 ≥ 90%。`)];
   if(j.online)contents.push(element('h3','90 秒在线回归'),table(['场景','跌倒','终点 (cm)','终点 1σ (cm)','终点 NEES','warn+alarm'],
     Object.entries(j.online).map(([n,o])=>[n,o.fell?'是':'否',cm(o.endpoint_xy_m),cm(o.endpoint_sigma_xy_m),o.endpoint_nees.toFixed(2),pct(o.quality.warn+o.quality.alarm)])));
   $('report').replaceChildren(...contents);
  }
 }catch{}
 setTimeout(reports,15000);
}
poll();reports();

function toggleHud(){const hidden=document.documentElement.classList.toggle('hud-hidden');$('hudToggle').textContent=hidden?'显示面板':'隐藏面板'}
$('hudToggle').onclick=toggleHud;
addEventListener('keydown',e=>{if((e.key==='h'||e.key==='H')&&!e.ctrlKey&&!e.metaKey&&!/INPUT|TEXTAREA/.test(document.activeElement.tagName))toggleHud()});
