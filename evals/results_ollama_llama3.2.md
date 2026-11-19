# Tool-calling accuracy eval

- Date: 2026-11-20
- Provider / model: `ollama` / `llama3.2`
- Prompt + tool-description hash: `31829de056`
- Suite(s): multi, single; 47 predictions over 1 pass(es)

**Overall accuracy: 27/47 (57.4%)**

| Suite | Accuracy |
|---|---|
| multi | 10/19 (52.6%) |
| single | 17/28 (60.7%) |

## Per-label precision / recall (all suites)

| Label | Precision | Recall | Support |
|---|---|---|---|
| cancel_reminder | 1.00 | 0.80 | 5 |
| list_reminders | 0.80 | 1.00 | 4 |
| none | 0.67 | 0.10 | 20 |
| search_web | 0.42 | 1.00 | 10 |
| set_reminder | 0.64 | 0.88 | 8 |

## Confusion matrix (rows = expected, columns = predicted)

| expected \ predicted | cancel_reminder | list_reminders | none | search_web | set_reminder |
|---|---|---|---|---|---|
| cancel_reminder | 4 | 0 | 0 | 0 | 1 |
| list_reminders | 0 | 4 | 0 | 0 | 0 |
| none | 0 | 1 | 2 | 14 | 3 |
| search_web | 0 | 0 | 0 | 10 | 0 |
| set_reminder | 0 | 0 | 1 | 0 | 7 |

## Failures (20)

| Suite | Case | Expected | Predicted |
|---|---|---|---|
| single | What's the capital of France? | none | search_web |
| single | Tell me a fun fact about octopuses | none | search_web |
| single | How do you say 'hello' in Spanish? | none | search_web |
| single | What's the boiling point of water? | none | search_web |
| single | Who wrote Pride and Prejudice? | none | search_web |
| single | What does photosynthesis mean? | none | search_web |
| single | Tell me a joke | none | search_web |
| single | How many days are in a leap year? | none | search_web |
| single | Give me a tip for sleeping better | none | search_web |
| single | Remind me to buy milk | none | set_reminder |
| single | Set a reminder | none | set_reminder |
| multi | text supplied after being asked: ...Call the dentist in an hour | set_reminder | none |
| multi | vague time is still not a time: ...Soon | none | set_reminder |
| multi | reminder request abandoned: ...Never mind | none | list_reminders |
| multi | change time = cancel first: ...Make it 30 minutes instead | cancel_reminder | set_reminder |
| multi | thanks after a search: ...Thanks, that's all I needed | none | search_web |
| multi | answerable from the search already done: ...Summarize that in one sentence | none | search_web |
| multi | interpretation needs no search: ...Is that warm or cold for October? | none | search_web |
| multi | another joke: ...Tell me another one | none | search_web |
| multi | translation follow-up: ...And in French? | none | search_web |
