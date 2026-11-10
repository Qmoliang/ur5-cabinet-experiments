const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('fs');
(async()=>{
 const base=process.env.ROOT_URL||'http://127.0.0.1:8767/';
 const browser=await chromium.launch({channel:'chrome',headless:true,args:['--enable-unsafe-swiftshader']});
 const page=await browser.newPage(),errors=[],bad=[],checks=[];
 page.on('pageerror',e=>errors.push(String(e)));page.on('response',r=>{if(r.status()>=400)bad.push([r.status(),r.url()]);});
 for(const width of [1440,390]){
  await page.setViewportSize({width,height:1000});
  for(const name of ['paper','ledger','explanation','tables']){
   await page.goto(base+'library/pages/liuqp900-'+name+'.html');
   await page.waitForFunction(()=>document.body.dataset.libraryReady==='true'&&(!document.querySelector('script[src*="mathjax-"]')||document.body.dataset.mathReady==='true'),{},{timeout:60000});
   await page.locator('article img').evaluateAll(async imgs=>{imgs.forEach(i=>i.loading='eager');await Promise.all(imgs.map(i=>i.decode()));});
   const state=await page.evaluate(()=>({width:document.documentElement.scrollWidth,viewport:innerWidth,math:document.querySelectorAll('mjx-container').length,mathErrors:document.querySelectorAll('mjx-merror,[data-mjx-error]').length,images:[...document.querySelectorAll('article img')].map(i=>({loaded:i.complete&&i.naturalWidth>0,width:i.clientWidth})),leakedMarkers:document.querySelector('article').innerText.includes('<!--')}));
   checks.push({name,...state});
   if(name==='paper'){
    await page.screenshot({path:`verification/paper-${width}.png`,fullPage:false});
    await page.getByRole('heading',{name:'8. 全部B组失败与关键退化',exact:true}).scrollIntoViewIfNeeded();
    await page.screenshot({path:`verification/paper-failures-${width}.png`,fullPage:false});
   }
  }
 }
 const result={base,checks,errors,bad};fs.writeFileSync('verification/paper-browser.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result));
 await browser.close();
 if(errors.length||bad.length||checks.some(x=>x.width>x.viewport||x.mathErrors||x.leakedMarkers||x.images.some(i=>!i.loaded)))process.exitCode=1;
})().catch(e=>{console.error(e);process.exit(1)});
