# Tool-calling accuracy eval

Provider: `ollama`

**Accuracy: 9/13 (69.2%)**

| Prompt | Expected | Predicted | Result |
|---|---|---|---|
| Remind me to call mom in 20 minutes | set_reminder | set_reminder | ✅ |
| Set a reminder to take the laundry out in 10 minutes | set_reminder | set_reminder | ✅ |
| Don't let me forget to submit my report in an hour | set_reminder | set_reminder | ✅ |
| What are my pending reminders? | list_reminders | list_reminders | ✅ |
| Do I have anything I asked you to remind me about? | list_reminders | list_reminders | ✅ |
| Search the web for the latest iPhone release | search_web | search_web | ✅ |
| What's the current weather in Tokyo right now? | search_web | search_web | ✅ |
| Look up who won the game last night | search_web | search_web | ✅ |
| What's the capital of France? | None | search_web | ❌ |
| What's 12 times 8? | None | None | ✅ |
| Tell me a fun fact about octopuses | None | search_web | ❌ |
| How do you say 'hello' in Spanish? | None | search_web | ❌ |
| What's the boiling point of water? | None | search_web | ❌ |
