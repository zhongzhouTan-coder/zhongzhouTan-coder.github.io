#!/usr/bin/env node

// Extraction flow adapted from Ademking/MD-This-Page at the pinned commit
// recorded below. The upstream MIT notice is preserved in
// third_party/MD-This-Page-LICENSE.txt.

import { createHash } from "node:crypto"
import {
  access,
  mkdir,
  readFile,
  readdir,
  rename,
  unlink,
  writeFile
} from "node:fs/promises"
import { isIP } from "node:net"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

import Defuddle from "defuddle"
import { JSDOM } from "jsdom"
import { chromium } from "playwright-core"
import TurndownService from "turndown"

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url))
const DEFAULT_ROOT = resolve(SCRIPT_DIR, "..")
const UPSTREAM_REFERENCE =
  "Ademking/MD-This-Page@dd8564584639c22bf083e6e3a10d7e84e6b379b5"
const EXTRACTOR = "defuddle@0.13.0 + turndown@7.2.4"
const DEFAULT_MAX_BYTES = 20 * 1024 * 1024
const PRIVATE_HOSTNAMES = new Set(["localhost", "localhost.localdomain"])
const SENSITIVE_RESPONSE_HEADERS = new Set([
  "set-cookie",
  "set-cookie2",
  "proxy-authenticate"
])

function fail(message) {
  throw new Error(message)
}

function usage() {
  return `Usage:
  node scripts/ingestion/web_to_markdown.mjs \\
    --url URL --category CATEGORY --slug SLUG [options]

Required:
  --url URL                 HTTP(S) page to capture.
  --category CATEGORY       Canonical category from kb-categories.json.
  --slug SLUG               Stable lowercase ASCII slug.

Options:
  --renderer MODE           auto (default), http, or chromium.
  --input-html PATH         Use an existing HTML file instead of downloading
                            it; useful for tests and manual browser exports.
  --docs-path PATH          Intended docs page; the source remains captured
                            until that page cites the snapshot.
  --browser-executable PATH Chromium executable for browser rendering.
  --timeout-ms NUMBER       Navigation/fetch timeout (default: 30000).
  --settle-ms NUMBER        Browser wait after DOMContentLoaded (default: 1000).
  --challenge-wait-ms N     Additional wait for automatic browser checks to
                            clear (default: 10000; does not solve CAPTCHAs).
  --max-bytes NUMBER        Maximum captured HTML bytes (default: 20971520).
  --min-content-chars N     Auto-render fallback threshold (default: 200).
  --allow-private-network   Permit explicit localhost/private IP URLs.
  --regenerate-derived      Rebuild Markdown/assets from an existing raw HTML
                            snapshot. Requires --input-html.
  --captured-at ISO_TIME    Override capture time (mainly for reproducible tests).
  --dry-run                 Capture and report paths without writing files.
  --root PATH               Knowledge-base root (defaults to repository root).
  --help                    Show this help.

Environment:
  WEB_INGEST_CHROMIUM       Default Chromium executable.
  WEB_INGEST_PROXY          Browser proxy URL; falls back to HTTPS_PROXY or
                            HTTP_PROXY. NO_PROXY is used as the bypass list.
`
}

function parseArgs(argv) {
  const options = {
    renderer: "auto",
    timeoutMs: 30_000,
    settleMs: 1_000,
    challengeWaitMs: 10_000,
    maxBytes: DEFAULT_MAX_BYTES,
    minContentChars: 200,
    allowPrivateNetwork: false,
    dryRun: false,
    root: DEFAULT_ROOT
  }
  const valueOptions = new Map([
    ["--url", "url"],
    ["--category", "category"],
    ["--slug", "slug"],
    ["--renderer", "renderer"],
    ["--input-html", "inputHtml"],
    ["--docs-path", "docsPath"],
    ["--browser-executable", "browserExecutable"],
    ["--timeout-ms", "timeoutMs"],
    ["--settle-ms", "settleMs"],
    ["--challenge-wait-ms", "challengeWaitMs"],
    ["--max-bytes", "maxBytes"],
    ["--min-content-chars", "minContentChars"],
    ["--captured-at", "capturedAt"],
    ["--root", "root"]
  ])
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index]
    if (argument === "--help") {
      options.help = true
      continue
    }
    if (argument === "--allow-private-network") {
      options.allowPrivateNetwork = true
      continue
    }
    if (argument === "--dry-run") {
      options.dryRun = true
      continue
    }
    if (argument === "--regenerate-derived") {
      options.regenerateDerived = true
      continue
    }
    const key = valueOptions.get(argument)
    if (!key) fail(`unknown argument: ${argument}`)
    const value = argv[index + 1]
    if (!value || value.startsWith("--")) fail(`${argument} requires a value`)
    options[key] = value
    index += 1
  }

  for (const key of [
    "timeoutMs",
    "settleMs",
    "challengeWaitMs",
    "maxBytes",
    "minContentChars"
  ]) {
    options[key] = Number(options[key])
    if (!Number.isInteger(options[key]) || options[key] < 0) {
      fail(`--${key.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`)} must be a non-negative integer`)
    }
  }
  options.root = resolve(options.root)
  if (options.inputHtml) options.inputHtml = resolve(options.inputHtml)
  return options
}

function validateOptions(options, categories) {
  for (const key of ["url", "category", "slug"]) {
    if (!options[key]) fail(`--${key} is required`)
  }
  if (!["auto", "http", "chromium"].includes(options.renderer)) {
    fail("--renderer must be auto, http, or chromium")
  }
  if (options.inputHtml && options.renderer === "chromium") {
    fail("--input-html cannot be combined with --renderer chromium")
  }
  if (options.regenerateDerived && !options.inputHtml) {
    fail("--regenerate-derived requires --input-html")
  }
  if (!categories[options.category]) {
    fail(`unknown category: ${options.category}`)
  }
  if (!categories[options.category].web_derived_prefix) {
    fail(`category has no web_derived_prefix: ${options.category}`)
  }
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(options.slug)) {
    fail("--slug must use lowercase ASCII letters, digits, and single hyphens")
  }
  if (options.docsPath) {
    const normalized = options.docsPath.replaceAll("\\", "/")
    const expectedPrefix = categories[options.category].docs_prefix
    if (
      normalized.startsWith("/") ||
      normalized.split("/").includes("..") ||
      !expectedPrefix ||
      !normalized.startsWith(expectedPrefix)
    ) {
      fail(`--docs-path must be beneath ${expectedPrefix || "the category docs prefix"}`)
    }
    options.docsPath = normalized
  }

  const parsedUrl = new URL(options.url)
  if (!["http:", "https:"].includes(parsedUrl.protocol)) {
    fail("--url must use http or https")
  }
  if (parsedUrl.username || parsedUrl.password) {
    fail("--url must not contain embedded credentials")
  }
  if (!options.allowPrivateNetwork && isExplicitPrivateHost(parsedUrl.hostname)) {
    fail(
      "localhost and private IP URLs are blocked; pass --allow-private-network to permit one"
    )
  }
  options.url = parsedUrl.href

  const captureTime = options.capturedAt ? new Date(options.capturedAt) : new Date()
  if (Number.isNaN(captureTime.valueOf())) fail("--captured-at must be an ISO timestamp")
  options.capturedAt = captureTime.toISOString()
}

function isExplicitPrivateHost(hostname) {
  const normalized = hostname.toLowerCase().replace(/^\[|\]$/g, "")
  if (PRIVATE_HOSTNAMES.has(normalized)) return true
  const ipVersion = isIP(normalized)
  if (ipVersion === 4) {
    const octets = normalized.split(".").map(Number)
    return (
      octets[0] === 10 ||
      octets[0] === 127 ||
      (octets[0] === 169 && octets[1] === 254) ||
      (octets[0] === 172 && octets[1] >= 16 && octets[1] <= 31) ||
      (octets[0] === 192 && octets[1] === 168)
    )
  }
  if (ipVersion === 6) {
    return normalized === "::1" || normalized.startsWith("fc") ||
      normalized.startsWith("fd") || normalized.startsWith("fe8") ||
      normalized.startsWith("fe9") || normalized.startsWith("fea") ||
      normalized.startsWith("feb")
  }
  return false
}

async function readLimitedBody(response, maxBytes) {
  if (!response.body) return ""
  const reader = response.body.getReader()
  const chunks = []
  let size = 0
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    size += value.byteLength
    if (size > maxBytes) {
      await reader.cancel()
      fail(`response exceeds --max-bytes (${maxBytes})`)
    }
    chunks.push(value)
  }
  const combined = new Uint8Array(size)
  let offset = 0
  for (const chunk of chunks) {
    combined.set(chunk, offset)
    offset += chunk.byteLength
  }
  return new TextDecoder().decode(combined)
}

function assertHtmlResponse(contentType, finalUrl) {
  const normalized = contentType.toLowerCase()
  if (normalized.includes("application/pdf") || finalUrl.toLowerCase().endsWith(".pdf")) {
    fail("URL resolves to a PDF; use the existing MinerU PDF workflow")
  }
  if (
    normalized &&
    !normalized.includes("text/html") &&
    !normalized.includes("application/xhtml+xml")
  ) {
    fail(`URL did not return HTML (content-type: ${contentType})`)
  }
}

function sanitizedHeaders(headers) {
  return Object.fromEntries(
    Object.entries(headers).filter(
      ([name]) => !SENSITIVE_RESPONSE_HEADERS.has(name.toLowerCase())
    )
  )
}

function accessChallengeReason(html, responseHeaders = {}) {
  if ((responseHeaders["cf-mitigated"] || "").toLowerCase() === "challenge") {
    return "the response was marked as a Cloudflare challenge"
  }

  const dom = new JSDOM(html)
  const { document } = dom.window
  const title = (document.title || "").replace(/\s+/g, " ").trim().toLowerCase()
  const text = (document.body?.textContent || "")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase()
  const knownChallengeElement = document.querySelector(
    [
      "#challenge-running",
      "#challenge-form",
      ".cf-challenge",
      "iframe[src*='challenges.cloudflare.com']",
      "script[src*='challenges.cloudflare.com']"
    ].join(",")
  )
  if (knownChallengeElement) {
    return "the page contains an automated-access verification challenge"
  }

  const captchaElement = document.querySelector(
    "[data-callback*='captcha'], .g-recaptcha, .h-captcha"
  )
  const challengeTitles = [
    "just a moment",
    "attention required",
    "security check",
    "verify you are human",
    "access denied"
  ]
  const challengePhrases = [
    "checking your browser",
    "verify you are human",
    "performing security verification",
    "enable javascript and cookies to continue",
    "unusual traffic from your computer network"
  ]
  const hasChallengeLanguage =
    challengeTitles.some((candidate) => title.includes(candidate)) ||
    challengePhrases.some((candidate) => text.includes(candidate))
  if (captchaElement && hasChallengeLanguage) {
    return "the page contains a human-verification challenge"
  }
  if (
    challengeTitles.some((candidate) => title.includes(candidate)) &&
    challengePhrases.some((candidate) => text.includes(candidate))
  ) {
    return `the page appears to be an access check (${document.title.trim()})`
  }
  return null
}

function assertNoAccessChallenge(capture) {
  const reason = accessChallengeReason(
    capture.html,
    capture.responseHeaders
  )
  if (reason) {
    fail(
      `${reason}; the capture was not saved. Retry with --renderer chromium, ` +
      "or complete the check in a normal browser and ingest an exported HTML " +
      "snapshot with --input-html. CAPTCHAs and access controls are not bypassed."
    )
  }
}

async function captureWithHttp(options) {
  if (options.inputHtml) {
    const html = await readFile(options.inputHtml, "utf8")
    if (Buffer.byteLength(html) > options.maxBytes) {
      fail(`input HTML exceeds --max-bytes (${options.maxBytes})`)
    }
    return {
      html,
      finalUrl: options.url,
      httpStatus: null,
      contentType: "text/html",
      renderer: "local-html",
      responseHeaders: {}
    }
  }

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs)
  try {
    const response = await fetch(options.url, {
      redirect: "follow",
      signal: controller.signal,
      headers: {
        "user-agent": "personal-docs-web-ingest/1.0"
      }
    })
    const finalUrl = response.url || options.url
    const contentType = response.headers.get("content-type") || ""
    assertHtmlResponse(contentType, finalUrl)
    if (!response.ok) fail(`HTTP ${response.status} while fetching ${finalUrl}`)
    const html = await readLimitedBody(response, options.maxBytes)
    return {
      html,
      finalUrl,
      httpStatus: response.status,
      contentType,
      renderer: "http",
      responseHeaders: sanitizedHeaders(
        Object.fromEntries(response.headers.entries())
      )
    }
  } finally {
    clearTimeout(timeout)
  }
}

async function firstAccessible(paths) {
  for (const path of paths) {
    if (!path) continue
    try {
      await access(path)
      return path
    } catch {
      // Try the next known executable.
    }
  }
  return null
}

function browserProxyConfig() {
  const value =
    process.env.WEB_INGEST_PROXY ||
    process.env.HTTPS_PROXY ||
    process.env.https_proxy ||
    process.env.HTTP_PROXY ||
    process.env.http_proxy
  if (!value) return undefined
  let proxyUrl
  try {
    proxyUrl = new URL(value)
  } catch {
    fail("browser proxy environment variable is not a valid URL")
  }
  if (!["http:", "https:", "socks5:"].includes(proxyUrl.protocol)) {
    fail(`unsupported browser proxy protocol: ${proxyUrl.protocol}`)
  }
  const proxy = {
    server: `${proxyUrl.protocol}//${proxyUrl.hostname}${proxyUrl.port ? `:${proxyUrl.port}` : ""}`
  }
  if (proxyUrl.username) proxy.username = decodeURIComponent(proxyUrl.username)
  if (proxyUrl.password) proxy.password = decodeURIComponent(proxyUrl.password)
  const bypass = process.env.NO_PROXY || process.env.no_proxy
  if (bypass) proxy.bypass = bypass
  return proxy
}

async function captureWithChromium(options) {
  const executablePath = await firstAccessible([
    options.browserExecutable,
    process.env.WEB_INGEST_CHROMIUM,
    process.env.CHROME_PATH,
    "/snap/bin/chromium",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable"
  ])
  if (!executablePath) {
    fail(
      "no Chromium executable found; pass --browser-executable or use --renderer http"
    )
  }

  const browser = await chromium.launch({
    executablePath,
    headless: true,
    proxy: browserProxyConfig()
  })
  try {
    const page = await browser.newPage()
    let latestDocumentResponse = null
    page.on("response", (candidate) => {
      if (
        candidate.request().resourceType() === "document" &&
        candidate.frame() === page.mainFrame()
      ) {
        latestDocumentResponse = candidate
      }
    })
    const navigationResponse = await page.goto(options.url, {
      waitUntil: "domcontentloaded",
      timeout: options.timeoutMs
    })
    if (options.settleMs) await page.waitForTimeout(options.settleMs)
    let html = await page.content()
    let response = latestDocumentResponse || navigationResponse
    let responseHeaders = response
      ? sanitizedHeaders(await response.allHeaders())
      : {}
    const challengeDeadline = Date.now() + options.challengeWaitMs
    while (
      accessChallengeReason(html, responseHeaders) &&
      Date.now() < challengeDeadline
    ) {
      await page.waitForTimeout(Math.min(500, challengeDeadline - Date.now()))
      html = await page.content()
      response = latestDocumentResponse || navigationResponse
      responseHeaders = response
        ? sanitizedHeaders(await response.allHeaders())
        : {}
    }
    const finalUrl = page.url()
    const contentType = (await response?.headerValue("content-type")) || "text/html"
    assertHtmlResponse(contentType, finalUrl)
    if (Buffer.byteLength(html) > options.maxBytes) {
      fail(`rendered page exceeds --max-bytes (${options.maxBytes})`)
    }
    const httpStatus = response?.status() ?? null
    if (httpStatus !== null && httpStatus >= 400) {
      fail(`HTTP ${httpStatus} while rendering ${finalUrl}`)
    }
    return {
      html,
      finalUrl,
      httpStatus,
      contentType,
      renderer: "chromium",
      responseHeaders
    }
  } finally {
    await browser.close()
  }
}

function makeAbsolute(document, selector, attribute, baseUrl) {
  for (const element of document.querySelectorAll(selector)) {
    const value = element.getAttribute(attribute)
    if (!value || value.startsWith("#") || value.startsWith("data:")) continue
    try {
      element.setAttribute(attribute, new URL(value, baseUrl).href)
    } catch {
      // Preserve malformed source values for the extractor to handle.
    }
  }
}

function normalizeImageSources(document, baseUrl) {
  const lazyAttributes = [
    "data-src",
    "data-lazy-src",
    "data-original",
    "data-url"
  ]
  for (const image of document.querySelectorAll("img")) {
    const currentSource = image.getAttribute("src") || ""
    const lazySource = lazyAttributes
      .map((attribute) => image.getAttribute(attribute))
      .find(Boolean)
    if (
      lazySource &&
      (!currentSource || currentSource.startsWith("data:image/svg+xml"))
    ) {
      image.setAttribute("src", lazySource)
    }
    const alt = image.getAttribute("alt")
    if (alt) image.setAttribute("alt", alt.replace(/\s+/g, " ").trim())
  }

  for (const source of document.querySelectorAll("source")) {
    if (!source.getAttribute("src") && source.getAttribute("data-src")) {
      source.setAttribute("src", source.getAttribute("data-src"))
    }
    if (!source.getAttribute("srcset") && source.getAttribute("data-srcset")) {
      source.setAttribute("srcset", source.getAttribute("data-srcset"))
    }
  }

  makeAbsolute(document, "img[src]", "src", baseUrl)
  makeAbsolute(document, "source[src]", "src", baseUrl)
  makeAbsolute(document, "image[href]", "href", baseUrl)
  for (const image of document.querySelectorAll("image")) {
    const value = image.getAttribute("xlink:href")
    if (!value || value.startsWith("#") || value.startsWith("data:")) continue
    try {
      image.setAttribute("xlink:href", new URL(value, baseUrl).href)
    } catch {
      // Preserve malformed source values in the derived SVG.
    }
  }
}

function rewriteSvgCssUrls(document, baseUrl) {
  for (const style of document.querySelectorAll("svg style")) {
    style.textContent = (style.textContent || "").replace(
      /url\(\s*(['"]?)([^'")]+)\1\s*\)/g,
      (match, quote, value) => {
        if (value.startsWith("#") || value.startsWith("data:")) return match
        try {
          return `url("${new URL(value, baseUrl).href}")`
        } catch {
          return match
        }
      }
    )
  }
}

function markdownAlt(value, fallback) {
  const normalized = (value || fallback)
    .replace(/\s+/g, " ")
    .replaceAll("[", "\\[")
    .replaceAll("]", "\\]")
    .trim()
  return normalized.length > 240
    ? `${normalized.slice(0, 237).trimEnd()}...`
    : normalized
}

function nearbySvgCaption(node) {
  let sibling = node.nextElementSibling
  if (!sibling && node.parentElement?.tagName?.toLowerCase() === "figure") {
    sibling = node.parentElement.querySelector("figcaption")
  }
  const text = sibling?.textContent?.replace(/\s+/g, " ").trim() || ""
  return /^(figure|diagram)\s*\d*/i.test(text) ? text : ""
}

function meaningfulSvg(node, caption) {
  if (node.getAttribute("aria-hidden") === "true") return false
  if (caption || node.querySelector("title")) return true
  const viewBox = (node.getAttribute("viewBox") || "")
    .trim()
    .split(/[\s,]+/)
    .map(Number)
  if (
    viewBox.length === 4 &&
    viewBox.every(Number.isFinite) &&
    (viewBox[2] >= 100 || viewBox[3] >= 100)
  ) {
    return true
  }
  return node.querySelectorAll("rect, circle, ellipse, path, line, polyline, polygon, text").length >= 5
}

function detachMeaningfulSvgs(document) {
  const detached = []
  for (const node of [...document.querySelectorAll("svg")]) {
    const caption = nearbySvgCaption(node)
    if (!meaningfulSvg(node, caption)) continue
    const token =
      `__WEB_INGEST_INLINE_SVG_${String(detached.length + 1).padStart(3, "0")}__`
    detached.push({ token, svg: node.outerHTML })
    const placeholder = document.createElement("p")
    placeholder.textContent = token
    node.replaceWith(placeholder)
  }
  return detached
}

function restoreMeaningfulSvgs(content, detached) {
  let restored = content
  for (const { token, svg } of detached) {
    const wrappedToken = new RegExp(
      `<([a-z][a-z0-9]*)[^>]*>\\s*${token}\\s*</\\1>`,
      "i"
    )
    if (wrappedToken.test(restored)) {
      restored = restored.replace(wrappedToken, svg)
    } else {
      restored = restored.replace(token, svg)
    }
  }
  return restored
}

function extractMarkdown(html, finalUrl) {
  const dom = new JSDOM(html, { url: finalUrl, contentType: "text/html" })
  const { document } = dom.window
  const canonicalElement = document.querySelector('link[rel~="canonical"][href]')
  let canonicalUrl = finalUrl
  if (canonicalElement) {
    try {
      canonicalUrl = new URL(canonicalElement.getAttribute("href"), finalUrl).href
    } catch {
      // Keep the final response URL when the canonical link is malformed.
    }
  }

  makeAbsolute(document, "a[href]", "href", finalUrl)
  normalizeImageSources(document, finalUrl)
  rewriteSvgCssUrls(document, finalUrl)
  const detachedSvgs = detachMeaningfulSvgs(document)

  const defuddle = new Defuddle(document, {
    url: finalUrl,
    removeExactSelectors: true
  })
  const result = defuddle.parse()
  let content = result?.content || ""
  if (!content.trim()) {
    document
      .querySelectorAll('script, style, link, noscript, svg, [aria-hidden="true"]')
      .forEach((element) => element.remove())
    const body =
      document.querySelector('[role="main"]') ||
      document.querySelector("main") ||
      document.querySelector("article") ||
      document.body
    content = body?.innerHTML || ""
  }
  content = restoreMeaningfulSvgs(content, detachedSvgs)

  const turndown = new TurndownService({
    bulletListMarker: "-",
    codeBlockStyle: "fenced",
    emDelimiter: "*"
  })
  const assets = []
  turndown.addRule("inline-svg-assets", {
    filter: (node) => node.nodeName.toLowerCase() === "svg",
    replacement: (_content, node) => {
      const caption = nearbySvgCaption(node)
      if (!meaningfulSvg(node, caption)) return ""
      const title = node.querySelector("title")?.textContent || ""
      const token = `__WEB_INGEST_ASSET_${String(assets.length + 1).padStart(3, "0")}__`
      let svg = node.outerHTML
      if (!/\sxmlns=/.test(svg)) {
        svg = svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"')
      }
      if (!svg.endsWith("\n")) svg += "\n"
      assets.push({
        token,
        mediaType: "image/svg+xml",
        extension: "svg",
        content: svg,
        alt: markdownAlt(caption || title, `Inline diagram ${assets.length + 1}`)
      })
      return `\n\n![${assets.at(-1).alt}](${token})\n\n`
    }
  })
  turndown.addRule("normalized-images", {
    filter: (node) => node.nodeName.toLowerCase() === "img",
    replacement: (_content, node) => {
      const source = node.getAttribute("src") || ""
      if (!source || source.startsWith("data:")) return ""
      const alt = markdownAlt(node.getAttribute("alt"), "Image")
      return `![${alt}](${source})`
    }
  })
  let markdown = turndown.turndown(content).trim()
  markdown = markdown
    .replace(/^[ \t]*[-·][ \t]*$/gm, "")
    .replace(/^[ \t]+$/gm, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim()
  if (!markdown) fail("extractor produced empty Markdown")

  return {
    markdown,
    title: result?.title || document.title || "",
    author: result?.author || "",
    publishedAt: result?.published || "",
    domain: result?.domain || new URL(finalUrl).hostname,
    canonicalUrl,
    assets
  }
}

async function captureAndExtract(options) {
  if (options.inputHtml) {
    const capture = await captureWithHttp(options)
    assertNoAccessChallenge(capture)
    return { capture, extraction: extractMarkdown(capture.html, capture.finalUrl) }
  }
  if (options.renderer === "http") {
    const capture = await captureWithHttp(options)
    assertNoAccessChallenge(capture)
    return { capture, extraction: extractMarkdown(capture.html, capture.finalUrl) }
  }
  if (options.renderer === "chromium") {
    const capture = await captureWithChromium(options)
    assertNoAccessChallenge(capture)
    return { capture, extraction: extractMarkdown(capture.html, capture.finalUrl) }
  }

  let httpError = null
  let httpCandidate = null
  try {
    const capture = await captureWithHttp(options)
    assertNoAccessChallenge(capture)
    const extraction = extractMarkdown(capture.html, capture.finalUrl)
    if (extraction.markdown.length >= options.minContentChars) {
      return { capture, extraction }
    }
    httpCandidate = { capture, extraction }
  } catch (error) {
    httpError = error
  }

  try {
    const capture = await captureWithChromium(options)
    assertNoAccessChallenge(capture)
    return { capture, extraction: extractMarkdown(capture.html, capture.finalUrl) }
  } catch (browserError) {
    if (httpCandidate) return httpCandidate
    const details = httpError ? `; HTTP capture failed: ${httpError.message}` : ""
    fail(`automatic Chromium fallback failed: ${browserError.message}${details}`)
  }
}

function yamlString(value) {
  return JSON.stringify(value ?? "")
}

function prepareDerivedAssets(extraction, webDerivedPrefix, snapshot) {
  let markdown = extraction.markdown
  const assetDirectory = `${snapshot}.assets`
  const assets = extraction.assets.map((asset, index) => {
    const sha256 = createHash("sha256").update(asset.content).digest("hex")
    const filename =
      `inline-${String(index + 1).padStart(2, "0")}-${sha256.slice(0, 12)}.${asset.extension}`
    const repositoryPath = `${webDerivedPrefix}${assetDirectory}/${filename}`
    markdown = markdown.replaceAll(
      asset.token,
      `${assetDirectory}/${filename}`
    )
    return {
      ...asset,
      sha256,
      repositoryPath,
      markdownPath: `${assetDirectory}/${filename}`
    }
  })
  return { markdown, assets }
}

function renderDerivedMarkdown(metadata, markdown, assets) {
  const assetMetadata = assets.length
    ? `assets:\n${assets.map((asset) => `  - ${yamlString(asset.repositoryPath)}`).join("\n")}\n`
    : ""
  return `---
kind: web-extraction
source_url: ${yamlString(metadata.requested_url)}
final_url: ${yamlString(metadata.final_url)}
canonical_url: ${yamlString(metadata.canonical_url)}
title: ${yamlString(metadata.title)}
author: ${yamlString(metadata.author)}
published_at: ${yamlString(metadata.published_at)}
captured_at: ${yamlString(metadata.captured_at)}
content_sha256: ${metadata.content_sha256}
renderer: ${metadata.renderer}
extractor: ${yamlString(metadata.extractor)}
${assetMetadata}---

${markdown}
`
}

function findSourcesArrayEnd(text) {
  const match = /"sources"\s*:\s*\[/.exec(text)
  if (!match) fail("sources.json has no sources array")
  let depth = 1
  let inString = false
  let escaped = false
  for (let index = match.index + match[0].length; index < text.length; index += 1) {
    const character = text[index]
    if (inString) {
      if (escaped) escaped = false
      else if (character === "\\") escaped = true
      else if (character === '"') inString = false
      continue
    }
    if (character === '"') inString = true
    else if (character === "[") depth += 1
    else if (character === "]") {
      depth -= 1
      if (depth === 0) return index
    }
  }
  fail("sources.json has an unterminated sources array")
}

function appendManifestEntry(text, entry) {
  const manifest = JSON.parse(text)
  const existing = manifest.sources.find((source) => source.id === entry.id)
  if (existing) {
    const immutableKeys = [
      "id",
      "title",
      "slug",
      "category",
      "kind",
      "source_url",
      "final_url",
      "revision",
      "raw_paths",
      "derived_path"
    ]
    for (const key of immutableKeys) {
      if (JSON.stringify(existing[key]) !== JSON.stringify(entry[key])) {
        fail(`manifest already contains conflicting ${key} for ${entry.id}`)
      }
    }
    return { text, reused: true }
  }

  const arrayEnd = findSourcesArrayEnd(text)
  const hasEntries = manifest.sources.length > 0
  const rendered = JSON.stringify(entry, null, 2)
    .split("\n")
    .map((line) => `    ${line}`)
    .join("\n")
  const insertion = `${hasEntries ? "," : ""}\n${rendered}\n  `
  return {
    text: text.slice(0, arrayEnd) + insertion + text.slice(arrayEnd),
    reused: false
  }
}

async function writeExclusive(path, content) {
  await mkdir(dirname(path), { recursive: true })
  await writeFile(path, content, { encoding: "utf8", flag: "wx" })
}

async function writeManifestAtomically(path, content) {
  const temporaryPath = `${path}.web-ingest-${process.pid}.tmp`
  await writeFile(temporaryPath, content, { encoding: "utf8", flag: "wx" })
  await rename(temporaryPath, path)
}

async function writeAtomically(path, content) {
  await mkdir(dirname(path), { recursive: true })
  const temporaryPath = `${path}.web-ingest-${process.pid}.tmp`
  await writeFile(temporaryPath, content, { encoding: "utf8", flag: "wx" })
  await rename(temporaryPath, path)
}

async function removeStaleDerivedAssets(root, assetDirectory, expectedPaths) {
  const absoluteDirectory = resolve(root, assetDirectory)
  let entries
  try {
    entries = await readdir(absoluteDirectory, { withFileTypes: true })
  } catch (error) {
    if (error.code === "ENOENT") return
    throw error
  }
  const expected = new Set(expectedPaths.map((path) => resolve(root, path)))
  for (const entry of entries) {
    if (!entry.isFile()) continue
    const absolutePath = resolve(absoluteDirectory, entry.name)
    if (!expected.has(absolutePath)) await unlink(absolutePath)
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2))
  if (options.help) {
    process.stdout.write(usage())
    return
  }

  const categoriesPath = resolve(options.root, "kb-categories.json")
  const manifestPath = resolve(options.root, "sources.json")
  const categories = JSON.parse(await readFile(categoriesPath, "utf8")).categories
  validateOptions(options, categories)

  const { capture, extraction } = await captureAndExtract(options)
  const contentSha256 = createHash("sha256").update(capture.html).digest("hex")
  const shortHash = contentSha256.slice(0, 12)
  const captureDate = options.capturedAt.slice(0, 10)
  const snapshot = `${options.slug}--web-${captureDate}-${shortHash}`
  const host = new URL(capture.finalUrl).hostname.toLowerCase()
  const sourceId = `web:${host}/${options.slug}@${captureDate}-${shortHash}`
  const rawHtmlPath = `raw/${options.category}/${snapshot}.html`
  const rawMetadataPath = `raw/${options.category}/${snapshot}.metadata.json`
  const derivedPath =
    `${categories[options.category].web_derived_prefix}${snapshot}.md`
  const prepared = prepareDerivedAssets(
    extraction,
    categories[options.category].web_derived_prefix,
    snapshot
  )

  const limitations = []
  if (capture.renderer === "local-html") {
    limitations.push(
      "HTML was supplied from a local file; HTTP status and response headers are unavailable."
    )
  }
  if (prepared.markdown.length < options.minContentChars) {
    limitations.push(
      `Extracted Markdown is shorter than ${options.minContentChars} characters.`
    )
  }
  const metadata = {
    schema_version: 1,
    requested_url: options.url,
    final_url: capture.finalUrl,
    canonical_url: extraction.canonicalUrl,
    title: extraction.title,
    author: extraction.author,
    published_at: extraction.publishedAt || null,
    captured_at: options.capturedAt,
    http_status: capture.httpStatus,
    content_type: capture.contentType,
    renderer: capture.renderer,
    content_sha256: contentSha256,
    raw_html_bytes: Buffer.byteLength(capture.html),
    extracted_markdown_characters: prepared.markdown.length,
    extracted_assets: prepared.assets.map((asset) => ({
      path: asset.repositoryPath,
      media_type: asset.mediaType,
      sha256: asset.sha256
    })),
    extractor: EXTRACTOR,
    upstream_reference: UPSTREAM_REFERENCE,
    response_headers: capture.responseHeaders,
    limitations
  }
  const derivedMarkdown = renderDerivedMarkdown(
    metadata,
    prepared.markdown,
    prepared.assets
  )
  const entry = {
    id: sourceId,
    title: extraction.title || options.slug,
    slug: options.slug,
    category: options.category,
    kind: "web",
    source_url: options.url,
    final_url: capture.finalUrl,
    captured_at: options.capturedAt,
    revision: contentSha256,
    raw_paths: [rawHtmlPath, rawMetadataPath],
    derived_path: derivedPath,
    ...(options.docsPath ? { docs_path: options.docsPath } : {}),
    status: "captured"
  }

  const manifestText = await readFile(manifestPath, "utf8")
  const manifestUpdate = appendManifestEntry(manifestText, entry)
  const report = {
    id: sourceId,
    renderer: capture.renderer,
    raw_html: rawHtmlPath,
    raw_metadata: rawMetadataPath,
    derived_markdown: derivedPath,
    derived_assets: prepared.assets.map((asset) => asset.repositoryPath),
    status: options.dryRun
      ? manifestUpdate.reused && options.regenerateDerived
        ? "dry-run-regenerate"
        : "dry-run"
      : manifestUpdate.reused
        ? options.regenerateDerived
          ? "regenerated"
          : "reused"
        : "captured"
  }
  if (options.dryRun) {
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`)
    return
  }
  if (manifestUpdate.reused) {
    if (options.regenerateDerived) {
      if (resolve(options.inputHtml) !== resolve(options.root, rawHtmlPath)) {
        fail(
          "--regenerate-derived input must be the manifest's immutable raw HTML path"
        )
      }
      const storedMetadata = JSON.parse(
        await readFile(resolve(options.root, rawMetadataPath), "utf8")
      )
      if (storedMetadata.content_sha256 !== contentSha256) {
        fail("raw metadata SHA-256 does not match the input HTML")
      }
      const regeneratedMetadata = {
        ...storedMetadata,
        extractor: EXTRACTOR
      }
      const regeneratedMarkdown = renderDerivedMarkdown(
        regeneratedMetadata,
        prepared.markdown,
        prepared.assets
      )
      for (const asset of prepared.assets) {
        await writeAtomically(
          resolve(options.root, asset.repositoryPath),
          asset.content
        )
      }
      await removeStaleDerivedAssets(
        options.root,
        `${categories[options.category].web_derived_prefix}${snapshot}.assets`,
        prepared.assets.map((asset) => asset.repositoryPath)
      )
      await writeAtomically(
        resolve(options.root, derivedPath),
        regeneratedMarkdown
      )
      process.stdout.write(`${JSON.stringify(report, null, 2)}\n`)
      return
    }
    for (const relativePath of [rawHtmlPath, rawMetadataPath, derivedPath]) {
      await access(resolve(options.root, relativePath))
    }
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`)
    return
  }

  const createdPaths = []
  try {
    const outputs = [
      [rawHtmlPath, capture.html],
      [rawMetadataPath, `${JSON.stringify(metadata, null, 2)}\n`],
      ...prepared.assets.map((asset) => [
        asset.repositoryPath,
        asset.content
      ]),
      [derivedPath, derivedMarkdown]
    ]
    for (const [relativePath, content] of outputs) {
      const absolutePath = resolve(options.root, relativePath)
      await writeExclusive(absolutePath, content)
      createdPaths.push(absolutePath)
    }
    await writeManifestAtomically(manifestPath, manifestUpdate.text)
  } catch (error) {
    for (const path of createdPaths.reverse()) {
      await unlink(path).catch(() => {})
    }
    throw error
  }
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`)
}

main().catch((error) => {
  process.stderr.write(`web ingest: ${error.message}\n`)
  process.exitCode = 1
})
