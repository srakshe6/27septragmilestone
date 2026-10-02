# Sample Q&A - MF facts assistant

Index built at: 2026-10-02T18:14:40+00:00  
Disclaimer shown in the UI: **Facts-only. No investment advice.**  
Scope: HDFC Mutual Fund - Large Cap, Flexi Cap, ELSS Tax Saver, Small Cap, Balanced Advantage (direct-growth variants).

Every answer below was produced by `python run.py samples` running the live pipeline; none of it is hand-written. Re-running the generator is byte-identical: the only time recorded is the index build, so a diff here means an answer, a citation or a guard changed. Latency is measured separately by `python bench.py`.

| # | Query | Outcome | Source |
|---|-------|---------|--------|
| 1 | What is the expense ratio of HDFC Large Cap Fund? | answered | [groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth](https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth) |
| 2 | minimum SIP for HDFC ELSS Tax Saver Fund | answered | [groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth](https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth) |
| 3 | Is there a lock-in period in HDFC ELSS Tax Saver Fund? | answered | [groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth](https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth) |
| 4 | What is the exit load on HDFC Balanced Advantage Fund? | answered | [groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth](https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth) |
| 5 | benchmark of HDFC Flexi Cap Fund | answered | [groww.in/mutual-funds/hdfc-equity-fund-direct-growth](https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth) |
| 6 | What is the riskometer level of HDFC Small Cap Fund? | answered | [www.hdfcmf.com/mutual-funds/equity/hdfc-large-cap-fund](https://www.hdfcmf.com/mutual-funds/equity/hdfc-large-cap-fund) |
| 7 | How do I download a capital gains statement? | answered | [www.camsonline.com/Service/Request](https://www.camsonline.com/Service/Request) |
| 8 | Where can I download my account statement? | answered | [investor.hdfcfund.com/](https://investor.hdfcfund.com/) |
| 9 | Should I buy HDFC Small Cap Fund? | refused (advice) | [www.hdfcmf.com/mutual-funds/equity/hdfc-large-cap-fund](https://www.hdfcmf.com/mutual-funds/equity/hdfc-large-cap-fund) |
| 10 | My PAN is ABCDE1234F, what is the exit load? | refused (PII) | [www.hdfcmf.com/mutual-funds/equity/hdfc-large-cap-fund](https://www.hdfcmf.com/mutual-funds/equity/hdfc-large-cap-fund) |

---

### 1. What is the expense ratio of HDFC Large Cap Fund?

*Cost figure, large cap*

```
The expense ratio of HDFC Large Cap Fund (Direct - Growth) is 1.04%.
Source: https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth
Last updated from sources: 2026-10-01
Facts-only. No investment advice.
```

Sentences: 1 &middot; Outcome: `answer` &middot; Guards: `G2_citation_ok`

### 2. minimum SIP for HDFC ELSS Tax Saver Fund

*Minimum investment, terse phrasing with no question mark*

```
The minimum SIP amount for HDFC ELSS Tax Saver Fund (Direct Plan - Growth) is ₹500 per month.
Source: https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth
Last updated from sources: 2026-10-01
Facts-only. No investment advice.
```

Sentences: 1 &middot; Outcome: `answer` &middot; Guards: `G2_citation_ok`

### 3. Is there a lock-in period in HDFC ELSS Tax Saver Fund?

*Statutory lock-in*

```
The lock-in period for HDFC ELSS Tax Saver Fund (Direct Plan - Growth) is 3 years. Units cannot be redeemed before the lock-in period ends.
Source: https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth
Last updated from sources: 2026-10-01
Facts-only. No investment advice.
```

Sentences: 2 &middot; Outcome: `answer` &middot; Guards: `G2_citation_ok`

### 4. What is the exit load on HDFC Balanced Advantage Fund?

*Non-standard exit-load wording (excess-units rule)*

```
The exit load of HDFC Balanced Advantage Fund (Direct - Growth) is 1% for units in excess of 15% of the investment, if redeemed within 1 year.
Source: https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth
Last updated from sources: 2026-10-01
Facts-only. No investment advice.
```

Sentences: 1 &middot; Outcome: `answer` &middot; Guards: `G2_citation_ok`

### 5. benchmark of HDFC Flexi Cap Fund

*Index benchmark, terse phrasing*

```
The benchmark of HDFC Flexi Cap Fund (Direct - Growth) is NIFTY 500 Total Return Index.
Source: https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth
Last updated from sources: 2026-10-01
Facts-only. No investment advice.
```

Sentences: 1 &middot; Outcome: `answer` &middot; Guards: `G2_citation_ok`

### 6. What is the riskometer level of HDFC Small Cap Fund?

*Fact not present in the quotable corpus - must point to the official factsheet*

```
That level is not in the source pages collected for this build. The SEBI riskometer level and the month-wise riskometer trend are published by the AMC in the scheme's monthly factsheet, available on the AMC's official scheme pages alongside the SID and KIM.
Source: https://www.hdfcmf.com/mutual-funds/equity/hdfc-large-cap-fund
Last updated from sources: 2026-10-03
Facts-only. No investment advice.
```

Sentences: 2 &middot; Outcome: `answer` &middot; Guards: `G2_citation_ok`

### 7. How do I download a capital gains statement?

*Procedural, routes to the RTA*

```
Download a capital gains / loss statement for your mutual fund holdings from the registrar's self-service request page, the Statements section of camsonline.com, where you choose the period, delivery option and fund houses to include. The statement is sent to the email address already registered with your folio, so no personal details need to be shared here.
Source: https://www.camsonline.com/Service/Request
Last updated from sources: 2026-10-03
Facts-only. No investment advice.
```

Sentences: 2 &middot; Outcome: `answer` &middot; Guards: `G2_citation_ok`

### 8. Where can I download my account statement?

*Procedural, routes to the AMC investor portal (a different destination)*

```
Download a mutual fund account or transaction statement from the AMC's official investor portal, investor.hdfcfund.com, using its statement request service to choose your date range. You sign in to your own account on the portal itself, so nothing personal needs to be shared here.
Source: https://investor.hdfcfund.com/
Last updated from sources: 2026-10-03
Facts-only. No investment advice.
```

Sentences: 2 &middot; Outcome: `answer` &middot; Guards: `G2_citation_ok`

### 9. Should I buy HDFC Small Cap Fund?

*Advice question - must refuse*

```
I can only share published facts about these schemes, so I can't say whether you should buy, sell or hold anything. The official scheme pages carry the factsheet, SID and KIM if you want to read the disclosures yourself.
Source: https://www.hdfcmf.com/mutual-funds/equity/hdfc-large-cap-fund
Last updated from sources: 2026-10-03
Facts-only. No investment advice.
```

Sentences: 2 &middot; Outcome: `advice_refusal` &middot; Guards: `stage5:advice_refusal, refusal_template`

### 10. My PAN is ABCDE1234F, what is the exit load?

*PII in the query - must refuse and not echo the PAN*

```
I don't accept or store PAN, Aadhaar, folio or account numbers, OTPs, email addresses, phone numbers, or UPI/IFSC bank details, so please don't send them. For anything account-specific, use the AMC investor portal or your RTA's own request page.
Source: https://www.hdfcmf.com/mutual-funds/equity/hdfc-large-cap-fund
Last updated from sources: 2026-10-03
Facts-only. No investment advice.
```

Sentences: 2 &middot; Outcome: `pii_refusal` &middot; Guards: `stage5:pii_refusal, refusal_template`

---

## Notes on these answers

- No answer recommends, ranks or compares schemes, and none states or computes a return.
- Every answer carries exactly one source link, drawn from the allow-list in `src/mf_faq/sources.py`.
- The riskometer query is answered by pointing at the AMC's monthly factsheet: the riskometer value is not present in the pages collected for this build, and the assistant says so rather than guessing.
- Query 10 is the PII case, and it is quoted verbatim here on purpose: the sample exists to document the refusal, which is not demonstrable without the triggering query. The assistant's reply never echoes the PAN, and the runtime redacts it before the query is logged or routed. So the PAN occurs in this file only in the query text above, and in no answer, citation or guard record.
