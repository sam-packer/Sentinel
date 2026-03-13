# Setup guide

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- PostgreSQL 17
- Twitter/X accounts with exported cookies (for scraping)

## 1. Install PostgreSQL 17

```bash
sudo apt install -y postgresql-common
sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh
sudo apt install -y postgresql-17
```

## 2. Create the database

```bash
sudo -u postgres psql
```

```sql
CREATE USER sentinel WITH PASSWORD 'your-secure-password';
CREATE DATABASE sentinel_db OWNER sentinel;
\q
```

No extensions required.

## 3. Install dependencies

```bash
uv sync
```

## 4. Configure

```bash
cp config.example.yaml config.yaml
cp .env.example .env
```

Edit `.env`:

```
DATABASE_URL=postgresql://sentinel:your-secure-password@localhost:5432/sentinel_db
```

## 5. Add Twitter accounts

Sentinel uses [twscrape](https://github.com/Telesphoreo/twscrape) for scraping. You need at least one Twitter/X account with exported cookies.

1. Log into Twitter/X in your browser
2. Export cookies as JSON using a browser extension (e.g., "Cookie-Editor")
3. Add the account:

```bash
uv run python add_account.py <twitter_username> <cookies.json>
```

Multiple accounts improve rate limit handling. Proxies can be configured in `config.yaml`:

```yaml
twitter:
  proxies:
    - socks5://user:pass@host1:port
    - socks5://user:pass@host2:port
```

## 6. Initialize and run

```bash
uv run setup
uv run collect -n 100
uv run serve
```

## Deployment

### Dedicated user setup

```bash
sudo useradd -m -s /bin/bash sentinel
sudo mkdir -p /opt/sentinel
sudo chown sentinel:sentinel /opt/sentinel
sudo -iu sentinel

curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo-url> /opt/sentinel/app
cd /opt/sentinel/app
cp config.example.yaml config.yaml
cp .env.example .env
editor .env
chmod 600 .env
uv sync
uv run setup
uv run python add_account.py <username> <cookies.json>
exit
```

### systemd

```bash
sudo cp /opt/sentinel/app/systemd/sentinel.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sentinel.service
```

## Environment variables

| Variable       | Required | Description                  |
|----------------|----------|------------------------------|
| `DATABASE_URL` | Yes      | PostgreSQL connection string |
