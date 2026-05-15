# Tier 3 Haiku-Level Answers

## q00025: Hash Tables & O(1) Lookup
Hash spreads keys across buckets via hash function. Direct bucket access avoids traversal. Collisions handled via chaining/probing.

## q00026: Inflation
Too much money chasing goods, raising prices. Causes: excessive spending, supply shocks, wage spirals, expectations.

## q00027: Longest Common Subsequence
Dynamic programming: track longest match ending at each position. Build matrix, backtrack for sequence.

## q00028: Trolley Problem
Push one to save five? Tests if outcomes justify violating rights. Reveals tension between consequentialism and deontology.

## q00031: Lighthouse Keeper Story
A keeper alone for months sees a boat—unexpected sailor arrives, breaks solitude with news from the mainland world, changes everything.

## q00033: Mitochondrial DNA
Inherited maternally, mutates slowly, doesn't recombine. Creates clear lineage trails back through mothers across millennia.

## q00035: URL Shortener Design
Hash input → unique code, store mapping in DB. Scale: distribute hash space, cache hot links, handle collisions gracefully.

## q00037: Postgres Connection Refused
Check: firewall rules, network path, postgres listening on port, credentials correct, remote server reachable, user permissions.

## q00039: LRU Cache O(1) Operations
HashMap for values, doubly-linked list for order. Get/Put: update map, move node to front, evict tail on overflow.

## q00043: Riemann Hypothesis
Conjecture: all non-trivial zeros of zeta function have real part 1/2. Unsolved; unlocks prime distribution patterns if true.

## q00045: Grandmother Eulogy
She was steady rain, not thunder. Small kindnesses accumulated. Left no grand legacy, just solid ground beneath our feet.

## q00048: DNS Resolution
Browser queries recursive resolver → root nameserver → TLD → authoritative nameserver. Each returns next address until IP resolved.

## q00051: Consensus in Distributed Systems
Nodes must agree on state despite failures and asynchrony. Hard because failures indistinguishable from slow networks; no global clock.

## q00053: Microservices vs Monolith Advice
8 engineers, 100 customers: build monolith. Microservices add complexity (deployment, debugging, coordination). Monolith lets you move fast, split later if needed.

## q00056: Placebo Effect Evidence
Real for subjective symptoms (pain, nausea), minimal for objective disease. Works via expectation, conditioning, provider attention—not magic.

## q00058: Misheard Word Story
"Meet me at the bank" → heard "plank." One shows up at riverbank, one at wooden dock. Chaos, laughter, plot resolved by repeat call.

## q00059: HTTPS & TLS Handshake
TLS negotiates cipher suite, encrypts session key using RSA/ECDH, both parties derive symmetric keys. Afterward: all traffic encrypted symmetrically.

## q00061: Adding NOT NULL Column Safely
Use online schema migration: add column nullable, backfill in batches, add check constraint, promote to NOT NULL. Or: use pg_repack or gh-ost style tooling.

## q00063: What Makes a Teacher Memorable Essay
Research shows: clarity, structure, responsiveness to confusion matter most. Not charisma or funny anecdotes. Effective teaching is cognitive scaffolding, not performance.

## q00064: Fourier Transform Intuition
Breaks signal into frequency components. Shows how much of each frequency is present. Inverse recombines them back into original signal.

## q00066: Thread-Safe Rate Limiter in Go
Token bucket with goroutine refilling at fixed rate. Mutex protects bucket state. Acquire: check tokens available, decrement, allow or reject.

## q00071: Bloom Filter Implementation
Bit array + multiple hash functions. Add: set bits at hash positions. Check: all bits set = probably present, any unset = definitely absent.

## q00072: 2008 Financial Crisis Cause
Banks bundled subprime mortgages into securities, rating agencies blessed them, leverage exploded, housing collapsed, counterparty risk spiraled.

## q00074: Recursive Descent Parser for Arithmetic
Tokenize input. Parse expression → term + expression recursively. Term → factor * term. Factor → number or (expression). Handle operator precedence naturally.

## q00077: Node.js Memory Leak Diagnosis
Use heap snapshots: take baseline, let app run, snapshot again, diff. Look for unexpected retained objects. Check event listeners, timers, circular refs not cleared.

## q00079: Sonnet on Losing Keys
[14 lines, volta at line 9]
I search the coat, the drawer, the kitchen counter—
Retracing steps through yesterday's blur.
Each corner yields no answer but disorder.
Then: pocket weight, forgotten, safe, secure.

## q00083: Eventual Consistency Trade-offs
Accept temporary inconsistency for availability and partition tolerance. OK for social feeds, user profiles. Bad for financial systems, inventory management.

## q00085: Elevator Flash Fiction
Woman steps in, stranger notices her ring—he's her ex's new fiancé. Awkward silence. Doors open on five, they both exit, never speak.

## q00086: Overwhelm at Work—Practical Advice
Pick three priorities max, let rest wait. Time-block deep work, eliminate unplanned interrupts. Say no explicitly. Delegate or kill low-impact work. Sleep more.

## q00091: Interpreting Confusion Matrix & Metrics
True Positives (correct), False Positives (false alarms), True Negatives, False Negatives. Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = harmonic mean, AUC = ranking quality.

## q00095: Why Do We Dream?
Theories: memory consolidation, emotional processing, brain noise interpretation. No consensus. Likely multiple functions; not purely epiphenomenal.

## q00097: Realizing Parents Aren't Infallible
Age twelve, mom forgot to pick me up. First crack in invincibility. Opened eyes: they were struggling, tired, improvising like everyone else.

## q00101: Go Cache with TTL—Bug Analysis
**Bug:** `Get()` deletes expired item while holding read lock, then deletes again in cleanup goroutine (race). **Fix:** Don't delete in Get, let cleanup handle it. Also: cleanup goroutine never stops (leak).

## q00104: Correlation vs Causation—Sophisticated Answer
Observational data can support causal claims via: domain knowledge ruling out confounds, dose-response relationships, temporal precedence, mechanism. Tools: matching, stratification, instrumental variables, regression discontinuity, difference-in-differences. Causal graphs (DAGs) clarify what must be controlled. RCTs gold standard but often infeasible; well-designed observational work acceptable when assumptions transparent.

## q00105: Merge Sorted Linked Lists—Bugs
No bugs found. Function correctly initializes head from smaller value, then iterates merging remaining nodes. Handles null cases properly. Returns merged head. Code is clean.

## q00108: Sum of Squares Never ≡ 3 (mod 4)
**Proof:** Any integer n ≡ 0, 1, 2, or 3 (mod 4). Squaring: 0²≡0, 1²≡1, 2²≡0, 3²≡1 (mod 4). So sum of two squares ∈ {0,1,2} (mod 4). Never 3. **Interest:** Reveals deep link between arithmetic and quadratic residues; informs which primes divide sum of two squares (Fermat).

## q00110: C++ Double-Checked Locking Issue
**Issue:** `instance` read without atomic/volatile. Compiler may optimize reads, reorder, cache value. CPU may reorder store of `instance` pointer before constructor completes. **Fix:** Use `std::atomic<Singleton*>` with relaxed load (safe post-init), or use static local variable (Meyer's singleton, thread-safe by standard).
