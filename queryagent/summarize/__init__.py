"""Summarisation backends.

  local   distilbart, offline, no key. Development default.
  hosted  Gemini or Groq. Required on serverless, where torch does not fit.

queryagent.pipeline picks between them from config.SUMMARIZER.
"""
