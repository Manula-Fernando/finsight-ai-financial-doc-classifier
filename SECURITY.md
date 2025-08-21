# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| >=0.1.0 | Yes       |

Older versions prior to the first public tag are not supported.

## Reporting a Vulnerability

1. **Do not open a public issue** if the vulnerability could expose sensitive data or allow remote execution.
2. Email: `SECURITY_CONTACT_PLACEHOLDER@example.com` with subject `FinSight AI Vulnerability` including:
   - Affected version / commit hash
   - Environment details (OS, Python version)
   - Steps to reproduce
   - Potential impact
3. You will receive acknowledgment within 72 hours. A coordinated disclosure timeline will be proposed.

## Disclosure Process
- Fix developed and tested privately.
- CVE filing (if applicable) coordinated with reporter.
- Public advisory & CHANGELOG entry upon release.

## Hardening Recommendations
- Run container as non-root (adjust Dockerfile to add a user for production).
- Use network isolation for internal deployments.
- Set environment variable `OCR_BACKEND` explicitly for deterministic behavior.
- Avoid uploading sensitive PII to public demo instances.
