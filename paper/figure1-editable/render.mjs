// Export the same editable SVG through Chromium, preserving text in the PDF.
import { createRequire } from 'node:module';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.BUNDLE_NODE_MODULES
  ? path.join(process.env.BUNDLE_NODE_MODULES, 'playwright') : 'playwright');
const dir = path.dirname(fileURLToPath(import.meta.url));
const svg = (await readFile(path.join(dir, 'OmniAnchor_editable.svg'), 'utf8'))
  .replace(/<\?xml[^>]*\?>/, '');
const [width, height] = svg.match(/viewBox="0 0 ([\d.]+) ([\d.]+)"/).slice(1).map(Number);
const browser = await chromium.launch({headless: true, channel: 'chrome'});
const page = await browser.newPage({viewport: {width, height}, deviceScaleFactor: 4});
await page.setContent(`<html><head><meta charset="utf-8"><style>
  @page { size: ${width}px ${height}px; margin: 0; }
  html,body { margin:0; width:${width}px; height:${height}px; background:#fff; }
  svg { display:block; }
  * { -webkit-print-color-adjust:exact; print-color-adjust:exact; }
  </style></head><body>${svg}</body></html>`);
await page.evaluate(async () => {
  await document.fonts.ready;
  await Promise.all([...document.querySelectorAll('image')].map(img => new Promise((ok, fail) => {
    const i = new Image(); i.onload = ok; i.onerror = fail;
    i.src = img.getAttribute('href') || img.getAttribute('xlink:href');
  })));
});
const validation = await page.evaluate(({width,height}) => {
  const text = [...document.querySelectorAll('svg text')].map(n => {
    const b = n.getBBox(); return {text:n.textContent.replace(/\u200b/g,''),x:b.x,y:b.y,width:b.width,height:b.height};
  });
  return {
    canvas: {width,height},
    editable_text_elements:text.length,
    images:document.querySelectorAll('svg image').length,
    rasterImagesEmbedded:[...document.querySelectorAll('svg image')].every(n => n.getAttribute('xlink:href').startsWith('data:image/')),
    vectorShapes:document.querySelectorAll('svg path, svg rect, svg line, svg circle').length,
    externalImageLinks:document.querySelectorAll('svg image[href^="http"]').length,
    outOfCanvasText:text.filter(b=>b.x < 0 || b.y < 0 || b.x+b.width > width || b.y+b.height > height),
    textBounds:text,
  };
}, {width,height});
await writeFile(path.join(dir, 'validation.json'), JSON.stringify(validation,null,2));
await page.screenshot({path:path.join(dir,'OmniAnchor_6144px.png'), animations:'disabled'});
await page.pdf({path:path.join(dir,'OmniAnchor_vector.pdf'),preferCSSPageSize:true,printBackground:true});
await page.close();
const preview = await browser.newPage({viewport:{width,height},deviceScaleFactor:1});
await preview.setContent(`<html><body style="margin:0">${svg}</body></html>`);
await preview.evaluate(() => document.fonts.ready);
await preview.screenshot({path:path.join(dir,'OmniAnchor_preview.png')});
await browser.close();
console.log(JSON.stringify({text:validation.editable_text_elements,shapes:validation.vectorShapes,
  embeddedPhotos:validation.images,overflow:validation.outOfCanvasText},null,2));
