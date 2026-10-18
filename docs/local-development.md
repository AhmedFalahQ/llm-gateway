# Local development

The whole gateway runs on your machine against a real DynamoDB, with no AWS
account, no credentials and no spend. Five commands:

```bash
make local-up      # start DynamoDB Local in Docker
make local-init    # create the tables, mint a dev key into .env.local
make local-test    # the full suite, against real DynamoDB instead of moto
make local-smoke   # send real requests through the gateway
make local-usage   # per-key spend, read back from the ledger
make local-down    # stop it
```

`make local-all` runs the middle four in order.

## Why bother, when moto already works

`make test` runs everything in process against moto, in a few seconds, with
nothing to install. That is the right default and it stays the default.

But a mock agrees with you more readily than the real service does, and the two
parts of this gateway most worth trusting are exactly the parts a mock can get
subtly wrong:

- **The conditional update** that makes the budget un-raceable. Its guarantee
  lives entirely in how DynamoDB evaluates `ConditionExpression` against a
  concurrent `ADD`. A mock that implements the happy path and returns the right
  exception type looks identical in a test and proves nothing.
- **The `Decimal` arithmetic** in the spend ledger. DynamoDB's number type has
  its own precision rules, and Python's `float` round-trips through them in ways
  that are easy to get away with until real money is involved.

So the same suite runs against both. One environment variable picks the backend:

| `AWS_ENDPOINT_URL_DYNAMODB` | Backend | Speed |
|---|---|---|
| unset | moto, in process | ~5 s |
| `http://localhost:8000` | DynamoDB Local over HTTP | ~7 s |

pytest prints which one it used in its header, so a run can never be ambiguous:

```
dynamodb backend: DynamoDB Local at http://localhost:8000
```

Nothing in `src/` changes between them. boto3 reads that variable itself, which
is why the production code has no idea the local path exists — and why there is
no `if local:` branch anywhere in the gateway.

## What the local loop shows that moto cannot

Run the smoke script twice:

```
$ make local-smoke && make local-usage
key                       spend    reqs    hits   hit %    tok in   tok out
dev                  $   0.0004       5       1     20%        48        91

$ make local-smoke && make local-usage
key                       spend    reqs    hits   hit %    tok in   tok out
dev                  $   0.0004      10       6     60%        48        91
```

Ten requests, but the spend did not move and the token counts did not move: the
entire second pass was served from cache. The ledger accumulated across two
separate processes, which is the behaviour that matters in production and the
one a per-test mock cannot demonstrate at all.

## Gotchas worth knowing before they cost you an hour

**`-sharedDb` is not optional.** Without it, DynamoDB Local partitions tables by
access key *and* region, so a table created with one set of dummy credentials is
invisible to a client using another. The symptom is
`ResourceNotFoundException` on a table you just watched get created. It is in
`docker-compose.yml`; leave it there.

**Credentials are pinned, not inherited.** Every local script and fixture passes
`local` / `local` explicitly instead of letting boto3 resolve a profile. Two
reasons: a script whose job is "create tables and mint a key" must not be able
to reach a real account because a profile happened to be exported, and DynamoDB
Local validates the *shape* of an access key — inheriting a placeholder from
some other tool fails as `UnrecognizedClientException`, which reads like a
permissions problem and is not one.

**`-inMemory` means every restart is a clean slate.** That is usually what you
want. For persistence, swap the command in `docker-compose.yml` for `-dbPath`
and uncomment the volume.

**`.env.local` holds the minted key** and is gitignored. `make local-init`
rewrites it only when it mints a new key, so re-running it will not silently
break `make local-smoke`.

## On WSL

Everything above works unchanged, with three setup details:

- Enable **WSL integration** in Docker Desktop's settings, or `docker compose`
  will not be reachable from the Linux shell.
- Keep the repository in the Linux filesystem (`~/projects/...`), not under
  `/mnt/c/`. Cross-filesystem I/O is dramatically slower and file watching does
  not work reliably.
- Open the folder with `code .` from inside WSL so the editor, terminal and
  Python interpreter all agree on which machine they are on.

`localhost:8000` resolves correctly from the WSL shell to a container published
by Docker Desktop, so no special host handling is needed.

## In CI

The `test-real-dynamodb` job runs `amazon/dynamodb-local` as a service container
and executes the same suite against it, on every push. The dual-backend claim is
therefore checked rather than asserted.

Both CI jobs run with **no secrets**. If either ever needs one, something has
leaked into the test path.
