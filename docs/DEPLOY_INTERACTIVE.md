# Deploying the interactive demo behind Cloudflare Access

The interactive demo (`tools/interactive_server.py`, `make interactive`) is a
**stateful server bound to the box**: it proxies chat to the local vllm-sr stack,
shells out to `docker`/`vllm-sr`, holds provider API keys, and can reload the
router. It therefore can **not** be a Cloudflare Worker (no Python/Docker/`:8899`
at the edge). The way to make it accessible is a **Cloudflare Tunnel** from the
box to a subdomain, with **Cloudflare Access** as the auth wall.

This is the **shared-router** model: several authenticated people chat at once
against the same well-configured router; each person's conversation history is
private to their browser; **Settings/Apply/Diagnostics are admin-only** (the
server gates them — see "Admin gate" below) so a viewer can't reload the shared
router or spend budget.

## 0. Prerequisites
- The box runs the demo locally: `make interactive` (serves `:8900`) with a
  healthy vllm-sr stack (`make route` etc.).
- A Cloudflare account with the `enterpriseai.center` zone.
- `cloudflared` installed on the box: <https://pkg.cloudflare.com/>.

## 1. Admin gate (do this first)
Set the admin allowlist so only you (and named operators) can mutate config /
Apply / read diagnostics; everyone else gets a read-only chat. Unset = open
(local dev only — never expose the port without this set).

```bash
export SR_ADMIN_EMAILS="you@corp.com,ops@corp.com"   # comma-separated
make interactive                                      # picks it up from the env
```

The verified identity comes from Cloudflare Access's
`Cf-Access-Authenticated-User-Email` header, which Access injects and strips from
client input. **Only trustworthy behind Access** — do not expose `:8900` raw.

## 2. Create the named tunnel
```bash
cloudflared tunnel login                     # browser auth, once
cloudflared tunnel create router-demo        # prints a TUNNEL-UUID + creds json
cp config/cloudflared.example.yml config/cloudflared.yml
# edit config/cloudflared.yml: set the TUNNEL-UUID, creds path, and hostname
cloudflared tunnel route dns router-demo router.enterpriseai.center
```

## 3. Run it
```bash
make interactive          # terminal 1 (with SR_ADMIN_EMAILS set)
make tunnel               # terminal 2 → cloudflared tunnel run, using config/cloudflared.yml
```
`router.enterpriseai.center` now reaches the box. (Run both under systemd /
`tmux` for persistence.)

## 4. Put Cloudflare Access in front (the auth wall)
In the Cloudflare dashboard → **Zero Trust → Access → Applications → Add (Self-hosted)**:
- **Application domain:** `router.enterpriseai.center`
- **Policy:** Allow → Include → *Emails* (your controlled list) or a domain / IdP group.
- Add the admin emails from step 1 to the allow policy too (Access controls *who
  can reach the site*; `SR_ADMIN_EMAILS` controls *who can change it* once in).

Now only allow-listed people can load the page, and only admins see Settings.

## 5. Usage telemetry (optional, needs the shell-worker side)
Two parts — both require the shell worker to accept the events:
- **Server-side** (rich): `interactive_server.py` POSTs chat events (routed tier,
  tokens, cost, the Access email) to the shell worker's usage endpoint with a
  shared token. *Not yet wired* — needs the endpoint URL + token + accepted
  schema from the worker repo.
- **Client-side** (page views): a browser beacon to
  `https://enterpriseai.center/api/public/usage-events`. Because the demo is on a
  **subdomain**, this is cross-origin → the shell worker must allow CORS for
  `router.enterpriseai.center` (or use a service token).

## Notes
- `config/cloudflared.yml` and `config/cloudflared.example.yml`: the former is
  gitignored (holds your tunnel id); commit only the example.
- Long requests (chat up to 180s) are fine over a tunnel.
- This is single-tenant: one router config, one stack. Concurrent **viewers** are
  fine; concurrent **admins** editing Settings would contend over one multi-minute
  Apply. For per-user *isolated* routers you'd need per-user vllm-sr stacks — a
  much larger build; see the assessment in the project history.
