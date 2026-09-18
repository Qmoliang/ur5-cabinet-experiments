const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs=require('fs');const path=require('path');
(async()=>{
 const root=path.resolve(__dirname,'..');const out=path.join(root,'verification');
 const browser=await chromium.launch({executablePath:process.env.BROWSER_EXECUTABLE || undefined,headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 const errors=[],requests=[];
 page.on('pageerror',e=>errors.push(String(e)));
 page.on('response',r=>{if(r.status()>=400)requests.push([r.status(),r.url()]);});
 await page.goto('http://127.0.0.1:8765');await page.waitForSelector('body[data-ready=true]');
 await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
 const checks={rows:await page.locator('#results-body tr').count()};
 await page.locator('#filter').selectOption('aborted');checks.aborted=await page.locator('#results-body tr').count();
 await page.locator('#filter').selectOption('all');
 await page.locator('#play').click();await page.waitForTimeout(1000);checks.play=await page.locator('#video-a').evaluate(v=>({time:v.currentTime,paused:v.paused,error:v.error}));
 await page.locator('#play').click();
 checks.media=[];
 for(const pair of ['known','online','historical','neo','ablation']){
  await page.locator('[data-pair="'+pair+'"]').click();
  for(const mode of ['real','cert']){
   await page.locator('[data-mode="'+mode+'"]').click();
   await page.waitForFunction(()=>document.querySelector('#video-a').readyState>=2 && (document.querySelector('#card-b').hidden || document.querySelector('#video-b').readyState>=2));
   await page.locator('#seek').evaluate(e=>{e.value=30;e.dispatchEvent(new Event('input',{bubbles:true}));});
   await page.waitForFunction(()=>[...document.querySelectorAll('video')].filter(v=>!v.closest('article').hidden).every(v=>!v.seeking && Math.abs(v.currentTime-5)<.05));
   checks.media.push({pair,mode,values:await page.locator('video').evaluateAll(vs=>vs.filter(v=>!v.closest('article').hidden).map(v=>({file:v.currentSrc.split('/').pop(),time:v.currentTime,error:v.error}))),time:await page.locator('#time').textContent()});
  }
 }
 for(const r of [5,20]){await page.locator('#radius').evaluate((e,v)=>{e.value=v;e.dispatchEvent(new Event('input',{bubbles:true}));},r);checks['radius'+r]=await page.locator('#layer-result').textContent();}
 const downloads=await page.locator('a[download], footer a').evaluateAll(es=>es.map(e=>e.getAttribute('href')).filter(h=>h&&!h.startsWith('#')));
 checks.downloads=[];
 for(const href of downloads){const res=await page.request.get('http://127.0.0.1:8765/'+href);checks.downloads.push({href,status:res.status()});}
 await page.locator('[data-pair="online"]').click();await page.locator('[data-mode="cert"]').click();
 await page.locator('#seek').evaluate(e=>{e.value=18;e.dispatchEvent(new Event('input',{bubbles:true}));});
 await page.waitForFunction(()=>[...document.querySelectorAll('video')].every(v=>!v.seeking && Math.abs(v.currentTime-3)<.05));await page.locator('#experiment').screenshot({path:path.join(out,'comparison.png')});
 await page.setViewportSize({width:390,height:844});await page.goto('http://127.0.0.1:8765');await page.waitForSelector('body[data-ready=true]');
 checks.mobile=await page.evaluate(()=>({viewport:innerWidth,document:document.documentElement.scrollWidth}));
 await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
 checks.errors=errors;checks.badRequests=requests;
 fs.writeFileSync(path.join(out,'browser-check.json'),JSON.stringify(checks,null,2));
 await browser.close();console.log(JSON.stringify(checks));
 if(checks.rows!==19||checks.aborted!==4||checks.play.paused||checks.play.time<=0||errors.length||requests.length||checks.mobile.document>checks.mobile.viewport)process.exitCode=1;
})().catch(e=>{console.error(e);process.exit(1);});

