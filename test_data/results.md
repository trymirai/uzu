# openai/privacy-filter on uzu — evaluation results

End-to-end smoke test of the ported `openai/privacy-filter` model on the `uzu` inference engine. Each example was run via `uzu classify` on the converted `.uzu` bundle (`/tmp/pf_uzu`, exported by `lalamo convert`, bf16 runtime) on Apple Metal.

- Model: **openai/privacy-filter** (20-layer MoE, 128 experts × top-4, GQA 14 heads / 2 kv-groups, YaRN RoPE, sliding window 128, attention sinks, bidirectional)
- Runtime: **uzu** · bf16 activations & weights · Metal backend
- Command: `./target/release/cli classify /tmp/pf_uzu --message "<sentence>"`
- Reproduce: `test_data/run_eval.sh` then `uv run python test_data/build_results_md.py`

## Aggregate performance

| metric | value |
|:---|---:|
| examples | **19** |
| total tokens classified | **365** |
| mean forward pass | **119.7** ms |
| median forward pass | **119.0** ms |
| min forward pass | **113.0** ms |
| max forward pass | **131.0** ms |
| mean total (incl. tokenize+post) | **119.7** ms |
| per-example throughput (mean) | **160.5** tok/s |
| amortized throughput | **160.5** tok/s |

`forward` is the model forward-pass time reported by uzu (prefill-only — classifier has no decode phase). `post` is post-processing (softmax + top-1 over labels). `total` includes tokenization, load-from-pool, etc.

## Per-example results

### Example 1

**Input:** Please send the receipt to john.doe@example.com tonight.

**Detected entities:** `john.doe@example.com` → **private_email**

**Stats:** `12` tokens · forward `113.0` ms · total `113.0` ms · `106.1` tok/s

<details>
<summary>Per-token predictions</summary>

| idx | token | label | conf |
|---:|:------|:------|-----:|
|   0 | `Please` | O | 1.0000 |
|   1 | `·send` | O | 1.0000 |
|   2 | `·the` | O | 1.0000 |
|   3 | `·receipt` | O | 1.0000 |
|   4 | `·to` | O | 1.0000 |
|   5 | `·john` | B-private_email | 0.9987 | ←
|   6 | `.d` | I-private_email | 0.9996 | ←
|   7 | `oe` | I-private_email | 0.9998 | ←
|   8 | `@example` | I-private_email | 0.9974 | ←
|   9 | `.com` | E-private_email | 0.9945 | ←
|  10 | `·tonight` | O | 0.9998 |
|  11 | `.` | O | 1.0000 |

</details>


### Example 2

**Input:** My phone number is +1 (415) 555-0173, call me after 6pm.

**Detected entities:** `+1 (415) 555-0173` → **private_phone**

**Stats:** `22` tokens · forward `115.0` ms · total `115.0` ms · `190.9` tok/s

<details>
<summary>Per-token predictions</summary>

| idx | token | label | conf |
|---:|:------|:------|-----:|
|   0 | `My` | O | 1.0000 |
|   1 | `·phone` | O | 1.0000 |
|   2 | `·number` | O | 1.0000 |
|   3 | `·is` | O | 1.0000 |
|   4 | `·+` | B-private_phone | 1.0000 | ←
|   5 | `1` | I-private_phone | 1.0000 | ←
|   6 | `·(` | I-private_phone | 1.0000 | ←
|   7 | `415` | I-private_phone | 1.0000 | ←
|   8 | `)` | I-private_phone | 1.0000 | ←
|   9 | `·` | I-private_phone | 1.0000 | ←
|  10 | `555` | I-private_phone | 1.0000 | ←
|  11 | `-` | I-private_phone | 1.0000 | ←
|  12 | `017` | I-private_phone | 1.0000 | ←
|  13 | `3` | E-private_phone | 1.0000 | ←
|  14 | `,` | O | 1.0000 |
|  15 | `·call` | O | 1.0000 |
|  16 | `·me` | O | 1.0000 |
|  17 | `·after` | O | 1.0000 |
|  18 | `·` | O | 1.0000 |
|  19 | `6` | O | 1.0000 |
|  20 | `pm` | O | 1.0000 |
|  21 | `.` | O | 1.0000 |

</details>


### Example 3

**Input:** I was born on March 14, 1987 in Palo Alto.

**Detected entities:** `March 14, 1987` → **private_date**

**Stats:** `15` tokens · forward `113.0` ms · total `113.0` ms · `133.3` tok/s

<details>
<summary>Per-token predictions</summary>

| idx | token | label | conf |
|---:|:------|:------|-----:|
|   0 | `I` | O | 1.0000 |
|   1 | `·was` | O | 1.0000 |
|   2 | `·born` | O | 1.0000 |
|   3 | `·on` | O | 1.0000 |
|   4 | `·March` | B-private_date | 1.0000 | ←
|   5 | `·` | I-private_date | 1.0000 | ←
|   6 | `14` | I-private_date | 1.0000 | ←
|   7 | `,` | I-private_date | 1.0000 | ←
|   8 | `·` | I-private_date | 1.0000 | ←
|   9 | `198` | I-private_date | 1.0000 | ←
|  10 | `7` | E-private_date | 1.0000 | ←
|  11 | `·in` | O | 1.0000 |
|  12 | `·Palo` | O | 1.0000 |
|  13 | `·Alto` | O | 1.0000 |
|  14 | `.` | O | 1.0000 |

</details>


### Example 4

**Input:** The wire transfer went to account 4567-8910-2345-6789 at Chase.

**Detected entities:** `4567-8910-2345-6789` → **account_number**

**Stats:** `21` tokens · forward `117.0` ms · total `117.0` ms · `179.2` tok/s

<details>
<summary>Per-token predictions</summary>

| idx | token | label | conf |
|---:|:------|:------|-----:|
|   0 | `The` | O | 1.0000 |
|   1 | `·wire` | O | 1.0000 |
|   2 | `·transfer` | O | 1.0000 |
|   3 | `·went` | O | 1.0000 |
|   4 | `·to` | O | 1.0000 |
|   5 | `·account` | O | 1.0000 |
|   6 | `·` | O | 1.0000 |
|   7 | `456` | B-account_number | 1.0000 | ←
|   8 | `7` | I-account_number | 1.0000 | ←
|   9 | `-` | I-account_number | 1.0000 | ←
|  10 | `891` | I-account_number | 1.0000 | ←
|  11 | `0` | I-account_number | 1.0000 | ←
|  12 | `-` | I-account_number | 1.0000 | ←
|  13 | `234` | I-account_number | 1.0000 | ←
|  14 | `5` | I-account_number | 1.0000 | ←
|  15 | `-` | I-account_number | 1.0000 | ←
|  16 | `678` | I-account_number | 1.0000 | ←
|  17 | `9` | E-account_number | 1.0000 | ←
|  18 | `·at` | O | 1.0000 |
|  19 | `·Chase` | O | 0.9999 |
|  20 | `.` | O | 1.0000 |

</details>


### Example 5

**Input:** Dr. Amelia Rodriguez signed the chart at 11:42am.

**Detected entities:** `Dr. Amelia Rodriguez` → **private_person**

**Stats:** `14` tokens · forward `115.0` ms · total `115.0` ms · `121.5` tok/s

<details>
<summary>Per-token predictions</summary>

| idx | token | label | conf |
|---:|:------|:------|-----:|
|   0 | `Dr` | I-private_person | 0.9627 | ←
|   1 | `.` | I-private_person | 0.9943 | ←
|   2 | `·Amelia` | I-private_person | 0.9195 | ←
|   3 | `·Rodriguez` | E-private_person | 0.9999 | ←
|   4 | `·signed` | O | 1.0000 |
|   5 | `·the` | O | 1.0000 |
|   6 | `·chart` | O | 1.0000 |
|   7 | `·at` | O | 1.0000 |
|   8 | `·` | O | 1.0000 |
|   9 | `11` | O | 0.9999 |
|  10 | `:` | O | 0.9999 |
|  11 | `42` | O | 0.9999 |
|  12 | `am` | O | 0.9999 |
|  13 | `.` | O | 1.0000 |

</details>


### Example 6

**Input:** Meet me at 221B Baker Street, London NW1 6XE on Friday.

**Detected entities:** `221B Baker Street, London NW1 6XE` → **private_address**

**Stats:** `18` tokens · forward `120.0` ms · total `120.0` ms · `149.6` tok/s

<details>
<summary>Per-token predictions</summary>

| idx | token | label | conf |
|---:|:------|:------|-----:|
|   0 | `Meet` | O | 1.0000 |
|   1 | `·me` | O | 1.0000 |
|   2 | `·at` | O | 1.0000 |
|   3 | `·` | O | 1.0000 |
|   4 | `221` | B-private_address | 0.9998 | ←
|   5 | `B` | I-private_address | 1.0000 | ←
|   6 | `·Baker` | I-private_address | 1.0000 | ←
|   7 | `·Street` | I-private_address | 0.9994 | ←
|   8 | `,` | I-private_address | 0.9998 | ←
|   9 | `·London` | I-private_address | 0.9987 | ←
|  10 | `·NW` | I-private_address | 0.9997 | ←
|  11 | `1` | I-private_address | 0.9887 | ←
|  12 | `·` | I-private_address | 0.9862 | ←
|  13 | `6` | I-private_address | 0.5455 | ←
|  14 | `XE` | E-private_address | 0.7130 | ←
|  15 | `·on` | O | 1.0000 |
|  16 | `·Friday` | O | 0.9999 |
|  17 | `.` | O | 0.9998 |

</details>


### Example 7

**Input:** The admin password is hunter2 — do not share it.

**Detected entities:** _(no PII detected)_

**Stats:** `12` tokens · forward `131.0` ms · total `131.0` ms · `91.7` tok/s

<details>
<summary>Per-token predictions</summary>

| idx | token | label | conf |
|---:|:------|:------|-----:|
|   0 | `The` | O | 1.0000 |
|   1 | `·admin` | O | 1.0000 |
|   2 | `·password` | O | 1.0000 |
|   3 | `·is` | O | 1.0000 |
|   4 | `·hunter` | O | 1.0000 |
|   5 | `2` | O | 1.0000 |
|   6 | `·âĢĶ` | O | 1.0000 |
|   7 | `·do` | O | 1.0000 |
|   8 | `·not` | O | 1.0000 |
|   9 | `·share` | O | 1.0000 |
|  10 | `·it` | O | 1.0000 |
|  11 | `.` | O | 1.0000 |

</details>


### Example 8

**Input:** Visit https://internal.corp.acme.io/reports/q4-2025 for the dashboard.

**Detected entities:** `https://internal.corp.acme.io/reports/q4-2025` → **private_url**

**Stats:** `20` tokens · forward `121.0` ms · total `121.0` ms · `165.0` tok/s

<details>
<summary>Per-token predictions</summary>

| idx | token | label | conf |
|---:|:------|:------|-----:|
|   0 | `Visit` | O | 0.9998 |
|   1 | `·https` | B-private_url | 0.9986 | ←
|   2 | `://` | I-private_url | 1.0000 | ←
|   3 | `internal` | I-private_url | 1.0000 | ←
|   4 | `.c` | I-private_url | 1.0000 | ←
|   5 | `orp` | I-private_url | 1.0000 | ←
|   6 | `.ac` | I-private_url | 1.0000 | ←
|   7 | `me` | I-private_url | 1.0000 | ←
|   8 | `.io` | I-private_url | 1.0000 | ←
|   9 | `/re` | I-private_url | 1.0000 | ←
|  10 | `ports` | I-private_url | 1.0000 | ←
|  11 | `/q` | I-private_url | 1.0000 | ←
|  12 | `4` | I-private_url | 1.0000 | ←
|  13 | `-` | I-private_url | 1.0000 | ←
|  14 | `202` | I-private_url | 1.0000 | ←
|  15 | `5` | E-private_url | 0.9997 | ←
|  16 | `·for` | O | 1.0000 |
|  17 | `·the` | O | 1.0000 |
|  18 | `·dashboard` | O | 1.0000 |
|  19 | `.` | O | 1.0000 |

</details>


### Example 9

**Input:** Hi Alex, the package will arrive at 42 Harrington Road, Cambridge, MA 02139.

**Detected entities:** `Alex` → **private_person**, `42 Harrington Road, Cambridge, MA 02139` → **private_address**

**Stats:** `21` tokens · forward `123.0` ms · total `123.0` ms · `171.2` tok/s

<details>
<summary>Per-token predictions</summary>

| idx | token | label | conf |
|---:|:------|:------|-----:|
|   0 | `Hi` | O | 1.0000 |
|   1 | `·Alex` | S-private_person | 1.0000 | ←
|   2 | `,` | O | 1.0000 |
|   3 | `·the` | O | 1.0000 |
|   4 | `·package` | O | 1.0000 |
|   5 | `·will` | O | 1.0000 |
|   6 | `·arrive` | O | 1.0000 |
|   7 | `·at` | O | 1.0000 |
|   8 | `·` | O | 1.0000 |
|   9 | `42` | B-private_address | 1.0000 | ←
|  10 | `·Harr` | I-private_address | 1.0000 | ←
|  11 | `ington` | I-private_address | 1.0000 | ←
|  12 | `·Road` | I-private_address | 1.0000 | ←
|  13 | `,` | I-private_address | 1.0000 | ←
|  14 | `·Cambridge` | I-private_address | 1.0000 | ←
|  15 | `,` | I-private_address | 1.0000 | ←
|  16 | `·MA` | I-private_address | 1.0000 | ←
|  17 | `·` | I-private_address | 1.0000 | ←
|  18 | `021` | I-private_address | 1.0000 | ←
|  19 | `39` | E-private_address | 1.0000 | ←
|  20 | `.` | O | 1.0000 |

</details>


### Example 10

**Input:** Charge card 5500 0000 0000 0004 expiring 08/28, CVV 737.

**Detected entities:** `08/28` → **private_date**

**Stats:** `26` tokens · forward `124.0` ms · total `124.0` ms · `209.6` tok/s

<details>
<summary>Per-token predictions</summary>

| idx | token | label | conf |
|---:|:------|:------|-----:|
|   0 | `Charge` | O | 1.0000 |
|   1 | `·card` | O | 1.0000 |
|   2 | `·` | O | 1.0000 |
|   3 | `550` | O | 1.0000 |
|   4 | `0` | O | 1.0000 |
|   5 | `·` | O | 1.0000 |
|   6 | `000` | O | 1.0000 |
|   7 | `0` | O | 1.0000 |
|   8 | `·` | O | 1.0000 |
|   9 | `000` | O | 0.9999 |
|  10 | `0` | O | 0.9999 |
|  11 | `·` | O | 1.0000 |
|  12 | `000` | O | 0.9998 |
|  13 | `4` | O | 0.9991 |
|  14 | `·exp` | O | 1.0000 |
|  15 | `iring` | O | 1.0000 |
|  16 | `·` | O | 1.0000 |
|  17 | `08` | B-private_date | 0.9235 | ←
|  18 | `/` | I-private_date | 0.8987 | ←
|  19 | `28` | E-private_date | 0.9318 | ←
|  20 | `,` | O | 0.9999 |
|  21 | `·CV` | O | 1.0000 |
|  22 | `V` | O | 1.0000 |
|  23 | `·` | O | 1.0000 |
|  24 | `737` | O | 1.0000 |
|  25 | `.` | O | 1.0000 |

</details>


### Example 11

**Input:** Her SSN 123-45-6789 was leaked in the breach last Tuesday.

**Detected entities:** `` → **account_number**, `123-45-6789` → **account_number**

**Stats:** `18` tokens · forward `125.0` ms · total `125.0` ms · `144.2` tok/s

<details>
<summary>Per-token predictions</summary>

| idx | token | label | conf |
|---:|:------|:------|-----:|
|   0 | `Her` | O | 0.9931 |
|   1 | `·SS` | O | 0.8720 |
|   2 | `N` | O | 0.4837 |
|   3 | `·` | I-account_number | 0.7546 | ←
|   4 | `123` | B-account_number | 0.5156 | ←
|   5 | `-` | I-account_number | 0.9999 | ←
|   6 | `45` | I-account_number | 1.0000 | ←
|   7 | `-` | I-account_number | 1.0000 | ←
|   8 | `678` | I-account_number | 1.0000 | ←
|   9 | `9` | E-account_number | 1.0000 | ←
|  10 | `·was` | O | 1.0000 |
|  11 | `·leaked` | O | 1.0000 |
|  12 | `·in` | O | 1.0000 |
|  13 | `·the` | O | 1.0000 |
|  14 | `·breach` | O | 1.0000 |
|  15 | `·last` | O | 1.0000 |
|  16 | `·Tuesday` | O | 1.0000 |
|  17 | `.` | O | 1.0000 |

</details>


### Example 12

**Input:** Bob Chen emailed bob.chen@startup.xyz about the merger on 2024-11-03.

**Detected entities:** `Bob Chen` → **private_person**, `bob.chen@startup.xyz` → **private_email**, `2024-11-03` → **private_date**

**Stats:** `21` tokens · forward `119.0` ms · total `119.0` ms · `176.6` tok/s

<details>
<summary>Per-token predictions</summary>

| idx | token | label | conf |
|---:|:------|:------|-----:|
|   0 | `Bob` | B-private_person | 0.9974 | ←
|   1 | `·Chen` | E-private_person | 0.9997 | ←
|   2 | `·emailed` | O | 0.9024 |
|   3 | `·bob` | B-private_email | 0.9603 | ←
|   4 | `.` | I-private_email | 0.9616 | ←
|   5 | `chen` | I-private_email | 0.9656 | ←
|   6 | `@` | I-private_email | 0.9884 | ←
|   7 | `startup` | I-private_email | 0.9862 | ←
|   8 | `.xyz` | E-private_email | 0.9883 | ←
|   9 | `·about` | O | 0.9997 |
|  10 | `·the` | O | 0.9994 |
|  11 | `·merger` | O | 0.9990 |
|  12 | `·on` | O | 1.0000 |
|  13 | `·` | O | 0.9999 |
|  14 | `202` | B-private_date | 1.0000 | ←
|  15 | `4` | I-private_date | 1.0000 | ←
|  16 | `-` | I-private_date | 1.0000 | ←
|  17 | `11` | I-private_date | 1.0000 | ←
|  18 | `-` | I-private_date | 0.9999 | ←
|  19 | `03` | E-private_date | 0.9999 | ←
|  20 | `.` | O | 1.0000 |

</details>


### Example 13

**Input:** Text Sarah at 415-867-5309 and Tom at tom@acme.co before noon.

**Detected entities:** `Sarah` → **private_person**, `415-867-5309` → **private_phone**, `Tom` → **private_person**, `tom@acme.co` → **private_email**

**Stats:** `21` tokens · forward `123.0` ms · total `123.0` ms · `170.7` tok/s

<details>
<summary>Per-token predictions</summary>

| idx | token | label | conf |
|---:|:------|:------|-----:|
|   0 | `Text` | O | 1.0000 |
|   1 | `·Sarah` | S-private_person | 1.0000 | ←
|   2 | `·at` | O | 1.0000 |
|   3 | `·` | O | 1.0000 |
|   4 | `415` | B-private_phone | 1.0000 | ←
|   5 | `-` | I-private_phone | 1.0000 | ←
|   6 | `867` | I-private_phone | 1.0000 | ←
|   7 | `-` | I-private_phone | 1.0000 | ←
|   8 | `530` | I-private_phone | 1.0000 | ←
|   9 | `9` | E-private_phone | 1.0000 | ←
|  10 | `·and` | O | 1.0000 |
|  11 | `·Tom` | S-private_person | 0.9996 | ←
|  12 | `·at` | O | 1.0000 |
|  13 | `·tom` | B-private_email | 1.0000 | ←
|  14 | `@` | I-private_email | 1.0000 | ←
|  15 | `ac` | I-private_email | 1.0000 | ←
|  16 | `me` | I-private_email | 1.0000 | ←
|  17 | `.co` | E-private_email | 0.9890 | ←
|  18 | `·before` | O | 0.9990 |
|  19 | `·noon` | O | 0.9994 |
|  20 | `.` | O | 1.0000 |

</details>


### Example 14

**Input:** The server at 10.0.42.17 logs requests under user_id=9f3c-alpha.

**Detected entities:** `10.0.42.17` → **private_url**, `user_id=9f3c-alpha` → **secret**

**Stats:** `23` tokens · forward `118.0` ms · total `118.0` ms · `195.0` tok/s

<details>
<summary>Per-token predictions</summary>

| idx | token | label | conf |
|---:|:------|:------|-----:|
|   0 | `The` | O | 1.0000 |
|   1 | `·server` | O | 1.0000 |
|   2 | `·at` | O | 1.0000 |
|   3 | `·` | O | 0.9999 |
|   4 | `10` | B-private_url | 0.9997 | ←
|   5 | `.` | I-private_url | 0.9999 | ←
|   6 | `0` | I-private_url | 0.9999 | ←
|   7 | `.` | I-private_url | 1.0000 | ←
|   8 | `42` | I-private_url | 0.9999 | ←
|   9 | `.` | I-private_url | 0.9999 | ←
|  10 | `17` | E-private_url | 0.6920 | ←
|  11 | `·logs` | O | 0.7838 |
|  12 | `·requests` | O | 0.9438 |
|  13 | `·under` | O | 0.9954 |
|  14 | `·user` | B-secret | 0.9441 | ←
|  15 | `_id` | I-secret | 0.8732 | ←
|  16 | `=` | I-secret | 0.9282 | ←
|  17 | `9` | I-secret | 0.9324 | ←
|  18 | `f` | I-secret | 0.9820 | ←
|  19 | `3` | I-secret | 0.9841 | ←
|  20 | `c` | I-secret | 0.9909 | ←
|  21 | `-alpha` | E-secret | 0.9829 | ←
|  22 | `.` | O | 1.0000 |

</details>


### Example 15

**Input:** No personal information in this sentence, just a friendly hello.

**Detected entities:** _(no PII detected)_

**Stats:** `12` tokens · forward `118.0` ms · total `118.0` ms · `101.7` tok/s

<details>
<summary>Per-token predictions</summary>

| idx | token | label | conf |
|---:|:------|:------|-----:|
|   0 | `No` | O | 1.0000 |
|   1 | `·personal` | O | 1.0000 |
|   2 | `·information` | O | 1.0000 |
|   3 | `·in` | O | 1.0000 |
|   4 | `·this` | O | 1.0000 |
|   5 | `·sentence` | O | 1.0000 |
|   6 | `,` | O | 1.0000 |
|   7 | `·just` | O | 1.0000 |
|   8 | `·a` | O | 1.0000 |
|   9 | `·friendly` | O | 1.0000 |
|  10 | `·hello` | O | 1.0000 |
|  11 | `.` | O | 1.0000 |

</details>


### Example 16

**Input:** Wire USD 12,340.00 to IBAN GB29 NWBK 6016 1331 9268 19 tomorrow.

**Detected entities:** `AN` → **account_number**, `GB29 NWBK 6016 1331 9268` → **account_number**

**Stats:** `28` tokens · forward `121.0` ms · total `121.0` ms · `232.1` tok/s

<details>
<summary>Per-token predictions</summary>

| idx | token | label | conf |
|---:|:------|:------|-----:|
|   0 | `Wire` | O | 0.9998 |
|   1 | `·USD` | O | 1.0000 |
|   2 | `·` | O | 1.0000 |
|   3 | `12` | O | 1.0000 |
|   4 | `,` | O | 1.0000 |
|   5 | `340` | O | 1.0000 |
|   6 | `.` | O | 1.0000 |
|   7 | `00` | O | 1.0000 |
|   8 | `·to` | O | 1.0000 |
|   9 | `·IB` | O | 0.7585 |
|  10 | `AN` | I-account_number | 0.4806 | ←
|  11 | `·GB` | B-account_number | 0.9397 | ←
|  12 | `29` | I-account_number | 1.0000 | ←
|  13 | `·NW` | I-account_number | 1.0000 | ←
|  14 | `BK` | I-account_number | 1.0000 | ←
|  15 | `·` | I-account_number | 0.9999 | ←
|  16 | `601` | I-account_number | 1.0000 | ←
|  17 | `6` | I-account_number | 1.0000 | ←
|  18 | `·` | I-account_number | 1.0000 | ←
|  19 | `133` | I-account_number | 1.0000 | ←
|  20 | `1` | I-account_number | 1.0000 | ←
|  21 | `·` | I-account_number | 1.0000 | ←
|  22 | `926` | I-account_number | 1.0000 | ←
|  23 | `8` | E-account_number | 0.9240 | ←
|  24 | `·` | O | 0.9778 |
|  25 | `19` | O | 0.9107 |
|  26 | `·tomorrow` | O | 0.9991 |
|  27 | `.` | O | 1.0000 |

</details>


### Example 17

**Input:** Alice Johnson lives at 1600 Pennsylvania Avenue NW and her DOB is 07/04/1990.

**Detected entities:** `Alice Johnson` → **private_person**, `1600 Pennsylvania Avenue NW` → **private_address**, `07/04/1990` → **private_date**

**Stats:** `22` tokens · forward `122.0` ms · total `122.0` ms · `180.6` tok/s

<details>
<summary>Per-token predictions</summary>

| idx | token | label | conf |
|---:|:------|:------|-----:|
|   0 | `Alice` | B-private_person | 0.9927 | ←
|   1 | `·Johnson` | E-private_person | 0.9999 | ←
|   2 | `·lives` | O | 1.0000 |
|   3 | `·at` | O | 1.0000 |
|   4 | `·` | O | 1.0000 |
|   5 | `160` | B-private_address | 0.9999 | ←
|   6 | `0` | I-private_address | 1.0000 | ←
|   7 | `·Pennsylvania` | I-private_address | 1.0000 | ←
|   8 | `·Avenue` | I-private_address | 0.9999 | ←
|   9 | `·NW` | E-private_address | 0.9998 | ←
|  10 | `·and` | O | 0.9998 |
|  11 | `·her` | O | 0.9998 |
|  12 | `·DOB` | O | 1.0000 |
|  13 | `·is` | O | 1.0000 |
|  14 | `·` | O | 1.0000 |
|  15 | `07` | B-private_date | 1.0000 | ←
|  16 | `/` | I-private_date | 1.0000 | ←
|  17 | `04` | I-private_date | 1.0000 | ←
|  18 | `/` | I-private_date | 1.0000 | ←
|  19 | `199` | I-private_date | 1.0000 | ←
|  20 | `0` | E-private_date | 0.9999 | ←
|  21 | `.` | O | 1.0000 |

</details>


### Example 18

**Input:** Copy API_KEY=sk-proj-abc123xyz789 into the .env file and restart.

**Detected entities:** `Copy API_KEY=sk-proj-abc123xyz789` → **secret**

**Stats:** `20` tokens · forward `119.0` ms · total `119.0` ms · `168.6` tok/s

<details>
<summary>Per-token predictions</summary>

| idx | token | label | conf |
|---:|:------|:------|-----:|
|   0 | `Copy` | I-secret | 0.6843 | ←
|   1 | `·API` | I-secret | 0.6775 | ←
|   2 | `_KEY` | I-secret | 0.6295 | ←
|   3 | `=` | I-secret | 0.7196 | ←
|   4 | `sk` | I-secret | 0.9378 | ←
|   5 | `-pro` | I-secret | 0.9877 | ←
|   6 | `j` | I-secret | 0.9968 | ←
|   7 | `-` | I-secret | 0.9980 | ←
|   8 | `abc` | I-secret | 0.9944 | ←
|   9 | `123` | I-secret | 0.9903 | ←
|  10 | `xyz` | I-secret | 0.9910 | ←
|  11 | `789` | E-secret | 0.9583 | ←
|  12 | `·into` | O | 0.9999 |
|  13 | `·the` | O | 1.0000 |
|  14 | `·.` | O | 1.0000 |
|  15 | `env` | O | 1.0000 |
|  16 | `·file` | O | 1.0000 |
|  17 | `·and` | O | 1.0000 |
|  18 | `·restart` | O | 1.0000 |
|  19 | `.` | O | 1.0000 |

</details>


### Example 19

**Input:** Forward the invoice to finance@globex.example and cc me at ops@globex.example.

**Detected entities:** `finance@globex.example` → **private_person**, `ops@globex.example` → **private_email**

**Stats:** `19` tokens · forward `117.0` ms · total `117.0` ms · `161.8` tok/s

<details>
<summary>Per-token predictions</summary>

| idx | token | label | conf |
|---:|:------|:------|-----:|
|   0 | `Forward` | O | 0.9991 |
|   1 | `·the` | O | 0.9999 |
|   2 | `·invoice` | O | 0.9998 |
|   3 | `·to` | O | 1.0000 |
|   4 | `·finance` | B-private_person | 0.9714 | ←
|   5 | `@` | I-private_person | 0.6967 | ←
|   6 | `glob` | I-private_person | 0.9970 | ←
|   7 | `ex` | I-private_person | 0.9971 | ←
|   8 | `.example` | E-private_person | 0.9947 | ←
|   9 | `·and` | O | 0.9998 |
|  10 | `·cc` | O | 1.0000 |
|  11 | `·me` | O | 1.0000 |
|  12 | `·at` | O | 1.0000 |
|  13 | `·ops` | B-private_email | 0.9878 | ←
|  14 | `@` | I-private_email | 0.9833 | ←
|  15 | `glob` | I-private_email | 0.9719 | ←
|  16 | `ex` | I-private_email | 0.9613 | ←
|  17 | `.example` | E-private_email | 0.6305 | ←
|  18 | `.` | O | 0.8413 |

</details>

