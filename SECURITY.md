# Security Policy

## Supported Versions

This project is maintained on a best-effort basis.

Security fixes are typically provided for:

- the latest stable release
- the current `main` branch (when feasible)

| Version | Supported |
| ------- | --------- |
| Latest stable release | ✅ |
| `main` | ✅ Best effort |
| Older releases | ❌ |

## Reporting a Vulnerability

If you believe you found a security issue, please **do not open a public issue**.

### Preferred method

Use **GitHub Security Advisories** (private vulnerability reporting) for this repository.

### Alternative method

If private reporting is not available, contact the maintainer privately (for example, via email listed on the GitHub profile or repository, if available).

## What to Include in the Report

Please include as much of the following as possible:

- A clear description of the issue
- Affected version, tag, or commit
- Steps to reproduce
- Expected behavior vs actual behavior
- Impact assessment (what an attacker can do)
- Environment details (OS, Python version, deployment type)
- Any suggested fix or mitigation (optional)

## Important: Do Not Share Secrets Publicly

This exporter may use sensitive local-network credentials and identifiers.

Please **never post publicly**:

- `local_key`
- device IDs
- internal IP addresses
- full `config.yaml` (unless fully redacted)
- private network topology details

If you need to share config snippets, redact all secrets and internal identifiers.

## Out of Scope (Generally)

The following are usually considered deployment or configuration issues, not code vulnerabilities, unless they result from unexpected behavior in the exporter itself:

- exposing the exporter port to the public Internet
- running without firewall or network restrictions
- publishing `/metrics` on untrusted networks
- committing secrets (`local_key`, device IDs, config files) to a public repository

If you are unsure, you can still report it privately and I will triage it.

## Response Expectations

This is a personal open-source project maintained on a best-effort basis.

Targets (not guarantees):

- Initial acknowledgment: within **7 days**
- Follow-up or triage update: as available
- Fix timeline: depends on severity and maintainer availability

## Disclosure Process

Please allow time for triage and a fix before public disclosure.

If a report is accepted, I may coordinate a fix and release before sharing details publicly.

## Bug Bounty

This project does not offer a bug bounty or paid vulnerability disclosure program.

## Scope Notes

This project is a **read-only Prometheus exporter** for Tuya smart plugs (local network polling via TinyTuya).

Typical security-relevant areas include:

- exposure of metrics endpoints (`/metrics`, health and readiness endpoints) on untrusted networks
- accidental leakage of secrets or identifiers in configuration or logs
- unsafe default deployment (no reverse proxy, no firewall, no authentication)
- denial-of-service or crash conditions caused by malformed or unexpected device responses

## Deployment Hardening Recommendations

For production or shared environments, consider:

- restricting network access to the exporter port (firewall, LAN-only)
- avoiding public Internet exposure
- placing the exporter behind a reverse proxy if remote access is required
- using network segmentation or VLANs for IoT devices
- storing config files with restricted permissions
- rotating credentials or keys if exposure is suspected
