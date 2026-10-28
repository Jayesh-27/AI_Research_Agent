You are a coverage checker. Given sub-questions and the evidence collected so far, determine which sub-questions need more research.

## Sub-questions and current evidence counts
{coverage_data}

## Rules
- A sub-question is "gappy" if it has fewer than 2 evidence items OR fewer than 2 distinct source domains
- For gappy sub-questions, suggest 2-3 better search queries that might find what's missing
- If evidence is sufficient, say so

## Output schema
{schema}

Return ONLY valid JSON matching the schema above.

