import re
import requests
from typing import Optional
from pydantic import Field
from langchain.prompts import PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models.llms import LLM
import pandas as pd
import os
import ast
import sqlparse
import keyword
import numpy as np
from langchain.schema import HumanMessage, AIMessage

# Custom LLM for making requests to OpenRouter API
class OpenRouterLLM(LLM):
    api_key: str = Field(...)
    model: str = Field(default="mistralai/mixtral-8x7b-instruct")

    def _call(self, prompt: str, stop: Optional[list] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        if "choices" not in data or not data["choices"]:
            print("OpenRouter API error response:", data)
            return f"⚠ OpenRouter API Error: {data.get('error', data)}"
        return data["choices"][0]["message"]["content"]

    @property
    def _llm_type(self) -> str:
        return "custom-openrouter"

# Main DB chain function
def get_db_chain(db_uri, api_key):
    db = SQLDatabase.from_uri(
        db_uri,
        include_tables=[
            "lego_sets",
            "lego_inventory_parts",
            "lego_parts",
            "lego_colors",
            "lego_themes",
            "lego_inventories",
            "lego_inventory_sets",
            "lego_part_categories"
        ]
    )
    llm = OpenRouterLLM(api_key=api_key)

    prompt = PromptTemplate(
        input_variables=["input", "table_info", "dialect"],
        template="""
You are a SQL assistant. Given the schema, return ONLY a valid SQL query that answers the user's question. Do NOT explain anything, do NOT include markdown, do NOT include any text before or after the SQL. Output ONLY the SQL query, nothing else.

- ALL table and column names MUST be double-quoted (e.g., "set_num", "lego_sets").
- Do NOT include any comments, explanations, or markdown.
- Do NOT include any text before or after the SQL query.
- Output ONLY the SQL query.
- When joining tables with overlapping column names, always use table aliases to qualify columns in the SELECT clause (e.g., p.part_num).
- Do NOT use reserved SQL keywords (such as is, in, on, as, by, etc.) as table aliases. Use safe aliases like inv_set, inv, inv_parts, etc.
- When asked for the year with the most records, group by the year column, count the records, and return the year with the highest count.
- The primary key for "lego_inventories" is "id". To find inventories with more than N parts, join "lego_inventories" (i) and "lego_inventory_parts" (ip) on i."id" = ip."inventory_id", group by i."id", and use HAVING COUNT(ip."part_num") > N.
- To count unique values, use COUNT(DISTINCT "column_name").
- When you need to aggregate data and join it to another table, use a subquery or CTE. For example, to get the top N sets with the most pieces, first count parts per set by joining "lego_inventory_parts" (ip) to "lego_inventories" (i) on ip."inventory_id" = i."id", then group by i."set_num". Then join the result to "lego_sets" (s) on s."set_num" = i."set_num".
- Example: To get the top 10 sets with the most pieces, use:
  SELECT s."set_num", s."name", s."year", s."theme_id", pc.num_parts
  FROM "lego_sets" s
  JOIN (
      SELECT i."set_num", COUNT(*) AS num_parts
      FROM "lego_inventory_parts" ip
      JOIN "lego_inventories" i ON ip."inventory_id" = i."id"
      GROUP BY i."set_num"
  ) pc ON s."set_num" = pc."set_num"
  ORDER BY pc.num_parts DESC
  LIMIT 10;
- If the user asks to "name", "list", "show", or "name it all" after a count, generate a SELECT query listing all names (e.g., SELECT "name" FROM "table_name"), not a count.
- For follow-up questions asking to list names, determine the appropriate table from context:
  * If asking about colors: SELECT "name" FROM "lego_colors"
  * If asking about themes: SELECT "name" FROM "lego_themes" 
  * If asking about sets: SELECT "name" FROM "lego_sets"
  * If asking about parts: SELECT "name" FROM "lego_parts"
- Always use the previous conversation context to resolve ambiguous or follow-up questions.
- If the user asks for a "vs." (versus) comparison, group by the relevant column (e.g., "is_trans" for transparent vs. non-transparent) and count the number of parts or usage for each group.
- Example: To compare transparent vs. non-transparent part usage, join "lego_inventory_parts" (ip) to "lego_parts" (p) and "lego_colors" (c), group by c."is_trans", and count the number of parts in each group:
  SELECT c."is_trans" AS "Transparent", COUNT(*) AS "Part Usage"
  FROM "lego_inventory_parts" ip
  JOIN "lego_parts" p ON ip."part_num" = p."part_num"
  JOIN "lego_colors" c ON ip."color_id" = c."id"
  GROUP BY c."is_trans";

Tables to use:
- lego_sets
- lego_inventory_parts
- lego_parts
- lego_colors
- lego_themes
- lego_inventories
- lego_inventory_sets
- lego_part_categories

SQL Dialect: {dialect}

Schema:
{table_info}

Question: "{input}"

SQL query:
"""
    )

    sql_chain = SQLDatabaseChain.from_llm(
        llm=llm,
        db=db,
        prompt=prompt,
        verbose=True,
        return_intermediate_steps=False,
        top_k=1,
        use_query_checker=True,
        return_sql=True
    )

    # Helper to extract SQL from LLM output
    def extract_sql(text):
        # Try to extract SQL from a code block first
        code_block = re.search(r"(?:sql)?\\s*([\\s\\S]+?)\\s*", text, re.IGNORECASE)
        if code_block:
            sql = code_block.group(1).strip()
        else:
            # Otherwise, extract the first SELECT ... ; statement
            select_stmt = re.search(r"(SELECT[\s\S]+?;)", text, re.IGNORECASE)
            if select_stmt:
                sql = select_stmt.group(1).strip()
            else:
                # Fallback: return the whole text
                sql = text.strip()

        sql = sql.replace("\\", "")
        sql = sql.replace("\\n", "\n")
        sql = sql.replace("\\", "\\")
        sql = sql.replace("\\", "")

        # Post-process to fix common subquery/CTE syntax errors
        def fix_subquery_syntax(sql):
            # Detect the pattern: ... GROUP BY ... ) SELECT ...
            match = re.search(r'(SELECT.+GROUP BY .+?\))\s*SELECT(.+)', sql, re.DOTALL)
            if match:
                subquery = match.group(1)
                main_select = 'SELECT' + match.group(2)
                fixed_sql = f"WITH part_counts AS (\n{subquery}\n)\n{main_select}"
                return fixed_sql
            return sql
        sql = fix_subquery_syntax(sql)

        # Fix reserved word aliases (e.g., IS) in SQL
        def fix_reserved_aliases(sql):
            # Replace 'JOIN ... IS ON' with 'JOIN ... inv_set ON'
            sql = re.sub(r'JOIN\s+"lego_inventory_sets"\s+IS\s+ON', 'JOIN "lego_inventory_sets" inv_set ON', sql, flags=re.IGNORECASE)
            # Replace 'is.' with 'inv_set.'
            sql = re.sub(r'\bis\.', 'inv_set.', sql)
            return sql
        sql = fix_reserved_aliases(sql)

        # Remove double closing parenthesis before SELECT
        sql = re.sub(r'\)\s*\)\s*SELECT', ') SELECT', sql)

        # Fix CTE name mismatches
        def fix_cte_name_mismatch(sql):
            # Find the CTE name after WITH
            match = re.search(r'WITH\s+([a-zA-Z0-9_]+)\s+AS', sql, re.IGNORECASE)
            if match:
                cte_name = match.group(1)
                # Replace all FROM <wrong_name> and JOIN <wrong_name> with FROM/JOIN <cte_name> if not already the CTE
                def replace_from_join(m):
                    keyword, table = m.group(1), m.group(2)
                    if table.lower() != cte_name.lower():
                        return f'{keyword} {cte_name}'
                    return m.group(0)
                sql = re.sub(r'(FROM|JOIN)\s+([a-zA-Z0-9_]+)', replace_from_join, sql, flags=re.IGNORECASE)
            return sql
        sql = fix_cte_name_mismatch(sql)

        # Fix CTE self-references (e.g., FROM part_counts in its own definition)
        def fix_cte_self_reference(sql):
            # If a CTE is defined as 'WITH part_counts AS (... FROM part_counts ...)', replace with the correct table
            match = re.search(r'WITH\s+part_counts\s+AS\s*\((.*?FROM\s+)(part_counts)(\s+\w+)', sql, re.IGNORECASE | re.DOTALL)
            if match:
                # Replace 'FROM part_counts' with 'FROM lego_colors'
                sql = re.sub(r'(WITH\s+part_counts\s+AS\s*\(.*?FROM\s+)(part_counts)(\s+\w+)', r'\1lego_colors\3', sql, flags=re.IGNORECASE | re.DOTALL)
            return sql
        sql = fix_cte_self_reference(sql)

        # Fix malformed SELECT ... GROUP BY ... ) SELECT ... FROM part_counts ...
        def wrap_select_groupby_as_cte(sql):
            match = re.search(r'^(SELECT.+GROUP BY .+?)\)\s*SELECT(.+FROM\s+part_counts.+)$', sql, re.DOTALL | re.IGNORECASE)
            if match:
                cte_body = match.group(1)
                main_select = 'SELECT' + match.group(2)
                fixed_sql = f"WITH part_counts AS (\n{cte_body}\n)\n{main_select}"
                return fixed_sql
            return sql
        sql = wrap_select_groupby_as_cte(sql)

        def fix_wrong_join_on_color_id(sql):
            # Replace wrong join on color_id with correct join on set_num
            sql = re.sub(r'pc\\.\"color_id\"\s*=\s*s\\.\"set_num\"', 'pc."set_num" = s."set_num"', sql)
            return sql
        sql = fix_wrong_join_on_color_id(sql)

        def fix_count_ls_id(sql):
            # Replace COUNT(ls."id") and COUNT(ls.id) with COUNT(*)
            sql = re.sub(r'COUNT\s*\(\s*ls\.\"id\"\s*\)', 'COUNT(*)', sql)
            sql = re.sub(r'COUNT\s*\(\s*ls\.id\s*\)', 'COUNT(*)', sql)
            return sql
        sql = fix_count_ls_id(sql)

        sql = fix_cte_name_mismatch(sql)

        def fix_missing_group_by(sql):
            # If SELECT ... COUNT(...) ... , <col> FROM ... ORDER BY ...; but no GROUP BY, add GROUP BY <col>
            match = re.match(
                r'SELECT\s+(.+COUNT\([^)]+\).+),\s*("?\w+"?)\s+FROM\s+([^\s;]+)(.*?);',
                sql, re.IGNORECASE | re.DOTALL
            )
            if match and "GROUP BY" not in sql.upper():
                count_part = match.group(1)
                name_col = match.group(2)
                from_table = match.group(3)
                rest = match.group(4)
                # Remove ORDER BY from rest if present, to append after GROUP BY
                order_by = ""
                rest_wo_order = rest
                order_match = re.search(r'(ORDER BY .+)', rest, re.IGNORECASE)
                if order_match:
                    order_by = order_match.group(1)
                    rest_wo_order = rest.replace(order_by, "")
                fixed_sql = f'SELECT {name_col}, {count_part} FROM {from_table}{rest_wo_order} GROUP BY {name_col} {order_by};'
                return fixed_sql
            return sql
        sql = fix_missing_group_by(sql)

        return sql

    class WrappedSQLChain:
        def __init__(self, chain, db, llm):
            self.chain = chain
            self.db = db
            self.llm = llm

        def run(self, question: str, chat_history=None):
            # Prepend chat history to the question for context
            previous_assistant_response = ""
            if chat_history:
                history_text = ""
                for msg in chat_history:
                    # Support both LangChain message objects and dicts
                    if hasattr(msg, "content"):
                        if isinstance(msg, HumanMessage):
                            history_text += f"User: {msg.content}\n"
                        elif isinstance(msg, AIMessage):
                            history_text += f"Assistant: {msg.content}\n"
                            previous_assistant_response = msg.content
                    elif isinstance(msg, dict):
                        if msg.get("role") == "user":
                            history_text += f"User: {msg.get('content', '')}\n"
                        elif msg.get("role") == "assistant":
                            history_text += f"Assistant: {msg.get('content', '')}\n"
                            previous_assistant_response = msg.get('content', '')
                question = f"{history_text}\nUser: {question}"
            print("Starting WrappedSQLChain.run")
            try:
                print("Before LLM call")
                raw_output = self.chain.run(question)
                print("After LLM call")
                debug_path = os.path.join(os.path.expanduser("~"), "debug_sql.txt")
                with open(debug_path, "a", encoding="utf-8") as f:
                    f.write(f"Raw LLM output: {repr(raw_output)}\n")
                print(f"Raw LLM output written to: {debug_path}")
                raw_output = raw_output.strip()
            except Exception as e:
                return f"⚠ LLM Chain Error:\n{e}"

            # Use the LLM-generated SQL, extracting it robustly
            sql_query_extracted = extract_sql(raw_output)
            # Post-processing fix for follow-up 'name' query after a count
            latest_question = question.split('\nUser:')[-1].strip() if 'User:' in question and isinstance(question, str) else question
            follow_up_detected = False
            if (
                isinstance(latest_question, str) and
                any(word in latest_question.lower() for word in ["name", "list", "show"]) and
                previous_assistant_response and isinstance(previous_assistant_response, str) and "count" in previous_assistant_response.lower()
            ):
                follow_up_detected = True
                # Determine which table to query based on context
                if "lego theme" in latest_question.lower() or "theme" in latest_question.lower():
                    sql_query_extracted = 'SELECT "name" FROM "lego_themes";'
                elif "lego color" in latest_question.lower() or "color" in latest_question.lower():
                    sql_query_extracted = 'SELECT "name" FROM "lego_colors";'
                elif "lego set" in latest_question.lower() or "set" in latest_question.lower():
                    sql_query_extracted = 'SELECT "name" FROM "lego_sets";'
                elif "lego part" in latest_question.lower() or "part" in latest_question.lower():
                    sql_query_extracted = 'SELECT "name" FROM "lego_parts";'
                else:
                    # Default to colors if no specific context
                    sql_query_extracted = 'SELECT "name" FROM "lego_colors";'
            if not (sql_query_extracted.lstrip().upper().startswith("SELECT") or sql_query_extracted.lstrip().upper().startswith("WITH")):
                return f"⚠ Invalid extracted SQL:\n\n{sql_query_extracted}"

            # Format SQL using sqlparse
            sql_query_extracted = sqlparse.format(sql_query_extracted, reindent=True, keyword_case='upper')

            debug_path = os.path.join(os.path.expanduser("~"), "debug_sql.txt")
            with open(debug_path, "a", encoding="utf-8") as f:
                f.write(f"SQL to execute:\n{sql_query_extracted}\n")
            print(f"Debug SQL written to: {debug_path}")

            with open(debug_path, "a", encoding="utf-8") as f:
                f.write(f"SQL bytes: {list(sql_query_extracted.encode())}\n")
                f.write(f"SQL length: {len(sql_query_extracted)}\n")

            print(f"Attempting SQL: {repr(sql_query_extracted)}")

            # Run SQL
            try:
                result = self.db.run(sql_query_extracted, include_columns=True)
                print(f"Raw SQL result: {repr(result)}")
                with open(debug_path, "a", encoding="utf-8") as f:
                    f.write(f"Raw SQL result: {repr(result)}\n")
                if not result:
                    return "✅ No results found."
                # If result is a string that looks like a list of dicts, parse it (robust to whitespace/newlines)
                if isinstance(result, str):
                    stripped = result.strip()
                    if stripped.startswith("[{") and stripped.endswith("}]"):
                        try:
                            # Replace Decimal('123') with 123 for safe parsing
                            stripped = re.sub(r"Decimal\('([0-9.]+)'\)", r"\1", stripped)
                            result = ast.literal_eval(stripped)
                        except Exception as e:
                            return f"⚠ Could not parse SQL result string: {e}\n\nRaw result: {repr(result)}"
                # --- Find previous user question and assistant response for follow-up context ---
                prev_user_q = None
                prev_assistant_resp = None
                if follow_up_detected and chat_history:
                    # Go backwards to find the last user and assistant messages before the latest user message
                    user_msgs = []
                    assistant_msgs = []
                    for msg in chat_history:
                        if hasattr(msg, 'content'):
                            if isinstance(msg, HumanMessage):
                                user_msgs.append(msg.content)
                            elif isinstance(msg, AIMessage):
                                assistant_msgs.append(msg.content)
                        elif isinstance(msg, dict):
                            if msg.get('role') == 'user':
                                user_msgs.append(msg.get('content', ''))
                            elif msg.get('role') == 'assistant':
                                assistant_msgs.append(msg.get('content', ''))
                    
                    # Debug logging
                    debug_path = os.path.join(os.path.expanduser("~"), "debug_sql.txt")
                    with open(debug_path, "a", encoding="utf-8") as f:
                        f.write(f"Follow-up detected. User messages: {len(user_msgs)}, Assistant messages: {len(assistant_msgs)}\n")
                        f.write(f"User messages: {user_msgs}\n")
                        f.write(f"Assistant messages: {assistant_msgs}\n")
                    
                    # Get the previous user question (the one before the current follow-up)
                    if len(user_msgs) >= 2:
                        prev_user_q = user_msgs[-2]  # Second-to-last user message
                    elif len(user_msgs) == 1:
                        prev_user_q = user_msgs[0]   # Only user message
                    
                    # Get the last assistant response
                    if assistant_msgs:
                        prev_assistant_resp = assistant_msgs[-1]
                    
                    # Debug logging
                    with open(debug_path, "a", encoding="utf-8") as f:
                        f.write(f"Extracted prev_user_q: {repr(prev_user_q)}\n")
                        f.write(f"Extracted prev_assistant_resp: {repr(prev_assistant_resp)}\n")
                # --- Compose output ---
                def format_followup_output(df):
                    parts = []
                    if prev_user_q and prev_assistant_resp:
                        parts.append(f"**Original Question:** {str(prev_user_q)}")
                        parts.append(f"**Original Answer:** {str(prev_assistant_resp)}")
                    parts.append(f"**Follow-up Question:** {str(latest_question)}")
                    parts.append(f"**Follow-up Answer:**\n{df.to_markdown(index=False)}")
                    
                    # Debug logging
                    debug_path = os.path.join(os.path.expanduser("~"), "debug_sql.txt")
                    with open(debug_path, "a", encoding="utf-8") as f:
                        f.write(f"Formatted output parts: {parts}\n")
                    
                    return ("\n\n".join(parts), df)
                # --- Return logic ---
                if isinstance(result, tuple) and len(result) == 2:
                    rows, headers = result
                    if not rows:
                        return "✅ No results found."
                    df = pd.DataFrame(rows, columns=headers)
                    category_col = None
                    value_col = None
                    for col in df.columns:
                        if category_col is None and (df[col].dtype == object or col.lower() == "year"):
                            category_col = col
                        if value_col is None and pd.api.types.is_numeric_dtype(df[col]) and col != category_col:
                            value_col = col
                    if follow_up_detected:
                        return format_followup_output(df)
                    else:
                        return (df.to_markdown(index=False), df)
                elif isinstance(result, list) and result and isinstance(result[0], dict):
                    df = pd.DataFrame(result)
                    category_col = None
                    value_col = None
                    for col in df.columns:
                        if category_col is None and (df[col].dtype == object or col.lower() == "year"):
                            category_col = col
                        if value_col is None and pd.api.types.is_numeric_dtype(df[col]) and col != category_col:
                            value_col = col
                    if follow_up_detected:
                        return format_followup_output(df)
                    else:
                        return (df.to_markdown(index=False), df)
                elif isinstance(result, list):
                    df = pd.DataFrame(result)
                    category_col = None
                    value_col = None
                    for col in df.columns:
                        if category_col is None and (df[col].dtype == object or col.lower() == "year"):
                            category_col = col
                        if value_col is None and pd.api.types.is_numeric_dtype(df[col]) and col != category_col:
                            value_col = col
                    if follow_up_detected:
                        return format_followup_output(df)
                    else:
                        return (df.to_markdown(index=False), df)
                else:
                    return f"⚠ Unexpected SQL result format: {repr(result)}"
            except Exception as e:
                with open(debug_path, "a", encoding="utf-8") as f:
                    f.write(f"DataFrame Error: {e}\n")
                    f.write(f"Raw result (in except): {repr(locals().get('result', None))}\n")
                return f"⚠ DataFrame Error: {e}\n\nRaw result: {repr(locals().get('result', None))}"

    return WrappedSQLChain(sql_chain,db,llm)