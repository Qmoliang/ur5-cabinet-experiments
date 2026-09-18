const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('fs'),path=require('path'),assert=require('assert');
(async()=>{
const root=path.resolve(__dirname,'..'),out=path.join(root,'verification','library');fs.mkdirSync(out,{recursive:true});
const base=process.env.SITE_URL||'http://127.0.0.1:8765/';
const manifest=JSON.parse(fs.readFileSync(path.join(root,'docs/library/manifest.json')));
const build=JSON.parse(fs.readFileSync(path.join(root,'docs/library/build-report.json')));
const browser=await chromium.launch({executablePath:process.env.BROWSER_EXECUTABLE,headless:true});
const page=await browser.newPage({viewport:{width:1440,height:1000}});
const errors=[],badRequests=[],results=[];page.on('pageerror',e=>errors.push(String(e)));page.on('response',r=>{if(r.status()>=400)badRequests.push([r.status(),r.url()]);});
await page.goto(base+'library/');await page.waitForSelector('body[data-library-ready=true]');
assert.equal(await page.locator('.doc-card:visible').count(),manifest.documents.length);
await page.locator('#doc-search').fill('NEO');assert.equal(await page.locator('.doc-card:visible').count(),4);
await page.locator('#doc-search').fill('');await page.locator('[data-filter="chapters"]').click();assert.equal(await page.locator('.doc-card:visible').count(),5);
await page.locator('[data-filter="all"]').click();await page.screenshot({path:path.join(out,'index-desktop.png'),fullPage:true});
for(const d of manifest.documents){
 await page.goto(base+'library/pages/'+d.id+'.html',{waitUntil:'domcontentloaded'});
 await page.waitForSelector('body[data-library-ready=true]');
 const expected=build.documents.find(x=>x.id===d.id);
 if(expected.math)await page.waitForSelector('body[data-math-ready=true]',{timeout:45000});
 if(expected.diagrams)await page.waitForFunction(()=>document.body.dataset.diagramsReady==='true'||document.body.dataset.diagramError,{},{timeout:45000});
 const result=await page.evaluate(()=>({equations:document.querySelectorAll('mjx-container').length,mathErrors:[...document.querySelectorAll('[data-mjx-error], [data-mml-node="merror"]')].map(x=>x.getAttribute('data-mjx-error')||x.textContent),diagramError:document.body.dataset.diagramError||null,diagrams:document.querySelectorAll('.mermaid svg').length,width:document.documentElement.scrollWidth,viewport:innerWidth}));
 result.id=d.id;result.expectedMath=expected.math;results.push(result);
 if(d.id==='ch02'){await page.screenshot({path:path.join(out,'chapter2-top.png')});await page.locator('.math-block').first().scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'chapter2-formulas.png')});}
 if(d.id==='ch03')await page.locator('.mermaid').first().screenshot({path:path.join(out,'chapter3-diagram.png')});
 console.log(JSON.stringify(result));
}
await page.setViewportSize({width:390,height:844});
for(const id of ['index','ch02','ch03']){
 await page.goto(base+'library/'+(id==='index'?'':'pages/'+id+'.html'));
 await page.waitForSelector('body[data-library-ready=true]');if(id!=='index')await page.waitForSelector('body[data-math-ready=true]');
 if(id==='ch03')await page.waitForSelector('body[data-diagrams-ready=true]');
 const widths=await page.evaluate(()=>({width:document.documentElement.scrollWidth,viewport:innerWidth}));results.push({id:'mobile-'+id,...widths});
 await page.screenshot({path:path.join(out,id+'-mobile.png')});
}
const download=await page.request.get(base+'library/'+manifest.documents.find(d=>d.format==='docx').file.split('/').map(encodeURIComponent).join('/'));
assert.equal(download.status(),200);assert.equal((await download.body()).length,manifest.documents.find(d=>d.format==='docx').bytes);
const report={results,errors,badRequests,wordDownload:download.status()};fs.writeFileSync(path.join(out,'browser-check.json'),JSON.stringify(report,null,2));
await browser.close();assert(!errors.length);assert(!badRequests.length);assert(results.every(r=>!r.mathErrors?.length&&!r.diagramError&&r.width<=r.viewport));console.log('Library browser checks passed');
})().catch(e=>{console.error(e);process.exit(1)});

