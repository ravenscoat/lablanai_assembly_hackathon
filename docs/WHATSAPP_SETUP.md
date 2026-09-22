# Setting up WhatsApp voice calling for the Hashim Girls Hostel agent

> **Goal:** People in Pakistan dial your WhatsApp Business number, the call rings
> on Meta's network (free for them — uses their data plan), and the LiveKit
> voice agent picks up and chats with them in Urdu.

> **Cost:** $0 to you for the first ~1,000 calls/month (Meta's free tier as of
> 2026). $0 to the caller (WhatsApp voice is over data, not PSTN).

## Prerequisites you need to do once (manual, on Meta's website)

### 1. Create a Meta Developer App (~5 minutes)

1. Go to https://developers.facebook.com/apps/ and click **Create app**.
2. **Use case:** "Other"
3. **App type:** "Business"
4. **App name:** `Hashim Girls Hostel Voice`
5. **Business portfolio:** create a new one if needed (this is the free
   "WhatsApp Business" workspace at https://business.facebook.com).
6. After the app is created, on the dashboard click **Add product** → find
   **WhatsApp** → click **Set up**.

### 2. Get a WhatsApp Business phone number (~5 minutes)

Inside the WhatsApp section of your app:

1. Click **API Setup**. Meta gives you a **free test phone number** for
   development — that's enough to start.
2. Copy:
   - **Phone number ID** (a long number like `123456789012345`)
   - **WhatsApp Business Account ID** (a different long number)
   - **Temporary access token** (24 h — we'll swap it for a permanent token
     in step 3)
3. Save these for the `.env` file later.

> When you're ready for production: in **WhatsApp → Phone numbers** click
> **Add phone number** and either port your existing PK number, or buy a new
> one. Either way Meta charges $0 for inbound calling on the WhatsApp Cloud
> API tier you're on.

### 3. Generate a permanent System User access token (~5 minutes)

The temporary token expires in 24 hours. Get a permanent one:

1. Go to https://business.facebook.com/settings → **Users → System users**
2. Click **Add** → name it `hashim-hostel-agent`, role: **Admin**
3. With the new system user selected, click **Generate new token**.
4. Pick your app, set **token expiration** to **Never**, and check the
   following permissions:
   - `whatsapp_business_messaging`
   - `whatsapp_business_management`
5. Copy the token. **This is your `WHATSAPP_API_KEY`** — keep it secret.

### 4. Subscribe the app to call webhooks (~3 minutes)

In your app's **WhatsApp → Configuration** page:

1. **Callback URL:** the public HTTPS URL of your webhook server (we set this
   up in the "Cloudflare Tunnel" step below). It must end in `/webhook`.
2. **Verify token:** any random string. Use the same value you put in
   `WHATSAPP_VERIFY_TOKEN` in `.env` (default in `.env.example` works fine).
3. Click **Verify and save** — Meta will hit your webhook with a GET request
   to confirm it works. Your webhook server returns the `hub.challenge`
   string and Meta accepts it.
4. Under **Webhook fields**, click **Manage** and subscribe to:
   - **calls** (the important one)
   - **messages** (optional, in case you want to also handle text later)
5. Also grab your **App Secret** from **App settings → Basic**. It's used to
   verify Meta's webhook signatures. Put it in `WHATSAPP_APP_SECRET` in
   `.env`.

### 5. Enable calling on the phone number (~2 minutes)

Inside your app's **WhatsApp → Phone numbers → [your number] → Settings**:

1. Find the **Calling** section, toggle **Allow incoming calls** ON.
2. Set call hours (24/7 is fine for a hostel).

## Local setup (after you've done the Meta steps above)

### 6. Fill in `.env`

The relevant block in `.env` (it's already templated):

```env
WHATSAPP_VERIFY_TOKEN=hashim-hostel-verify-2026-change-me  # whatever you put in Meta UI
WHATSAPP_APP_SECRET=<paste-from-App-Settings-Basic>
WHATSAPP_API_KEY=<paste-system-user-permanent-token>
WHATSAPP_PHONE_NUMBER_ID=<paste-from-API-Setup>
WHATSAPP_CLOUD_API_VERSION=23.0
WHATSAPP_DESTINATION_COUNTRY=PK
WHATSAPP_WEBHOOK_PORT=8090
```

### 7. Start the webhook server

```bash
cd "/Users/mudassar/Desktop/hashim girls hostel voice agent/urdu-voice-agent"
PYTHONPATH=src .venv/bin/python scripts/whatsapp_webhook.py
```

You should see:

```
WhatsApp → LiveKit bridge listening on :8090 (agent=hashim-girls-hostel-agent, wa_phone_id=...)
```

### 8. Expose it to the internet with a Cloudflare Tunnel (free)

Meta needs a public HTTPS URL to call. The easiest free option is Cloudflare
Tunnel — no account, no domain needed, gives you a `*.trycloudflare.com` URL
in 10 seconds.

```bash
brew install cloudflared             # one-time
cloudflared tunnel --url http://localhost:8090
```

It prints a URL like:

```
https://wild-frogs-jump-quickly.trycloudflare.com
```

**Paste that URL + `/webhook` into Meta's Callback URL** field (step 4 above).

> For something more stable later: use ngrok, a paid Cloudflare named tunnel,
> or deploy this script to Fly.io / Render.com (also free tier).

### 9. Make a test call

1. Save the Meta-assigned phone number as a contact in your WhatsApp (the
   phone number is shown in the **API Setup** page).
2. Tap the contact, then the **call** button.
3. Watch the `whatsapp_webhook.py` logs — you'll see:

```
wa call event=connect id=wacid.ABGGFjFVU...
accepting WhatsApp call wacid.ABGGFjFVU... from <YOUR_NUMBER>
✅ call wacid.ABGGFjFVU... accepted; room=... agent=hashim-girls-hostel-agent caller=<YOUR_NUMBER>
```

4. Your phone connects, you hear the agent greet in Urdu, you can talk
   normally — done.

## Architecture

```
WhatsApp user dials
   ┌──────────────┐
   │  Caller phone │
   └──────┬───────┘
          │ WhatsApp voice (over data)
          ▼
   ┌──────────────┐
   │  Meta Cloud   │  ← sends "calls.connect" webhook with SDP offer
   │   WhatsApp    │ ───┐
   └──────────────┘    │ HTTPS POST /webhook
                       ▼
              ┌────────────────────────┐
              │ whatsapp_webhook.py    │  scripts/whatsapp_webhook.py
              │   (Cloudflare Tunnel) │
              └────────┬───────────────┘
                       │ accept_whatsapp_call(SDP, agents=[hashim-girls-hostel-agent])
                       ▼
              ┌────────────────────────┐
              │ LiveKit Connector API  │  pre-accepts WhatsApp call,
              │                       │  forwards SDP answer,
              │                       │  spawns LiveKit room
              └────────┬───────────────┘
                       │ dispatch
                       ▼
              ┌────────────────────────┐
              │ hashim-girls-hostel-   │  Speechmatics STT →
              │     agent              │  Gemini LLM →
              │  (already running)     │  Azure ur-PK-UzmaNeural TTS
              └────────────────────────┘
                       │ audio
                       ▼
                   Caller hears
                  "السلام علیکم!"
```

## Cost summary (as of June 2026)

| Item | Cost |
|------|------|
| WhatsApp Business Cloud API — first ~1,000 marketing-initiated calls/month | **Free** |
| User-initiated calls (people calling your business) | **Free** at any volume |
| Test phone number from Meta | **Free** (use forever for dev) |
| Bringing your own number to WhatsApp Business | **Free**, takes ~24h verification |
| Cloudflare Tunnel | **Free** |
| Caller's side | **Free** (data only — no PSTN charges) |
| LiveKit Cloud | Your existing plan (no extra for connector) |

## Troubleshooting

- **"Verify and save" fails in Meta UI** → the webhook server isn't reachable.
  Confirm `cloudflared` is running and the URL ends in `/webhook`.
- **Webhook delivers but no audio** → check `WHATSAPP_API_KEY` /
  `WHATSAPP_PHONE_NUMBER_ID` are set, agent is running, and
  `accept_whatsapp_call` returned a room name in the logs.
- **Audio one-way** → make sure `WHATSAPP_DESTINATION_COUNTRY=PK` so Meta
  routes media to its India/Asia cluster (lower latency for PK callers).
- **Call rejected immediately** → check `WHATSAPP_APP_SECRET` and the
  `X-Hub-Signature-256` validation in the logs.

## Rotating credentials when you're done testing

- **System User access token:** Meta Business Settings → Users → System users
  → click your token → **Revoke**
- **App secret:** Meta App Dashboard → App settings → Basic → **Reset**
