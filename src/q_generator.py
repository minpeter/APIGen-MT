import random
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from tool_manager import ToolManager
from llm_client import LLMClient
from sentence_transformers import SentenceTransformer, util
import torch
import heapq
import json
import os


class QGenerator:
    def __init__(
        self,
        tool_manager: ToolManager,
        llm: LLMClient,
        sentence_model_id: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    ):
        raw_tools = tool_manager.get_tools()
        self.tool_schemas_json_str = json.dumps(raw_tools, indent=2) # Store as JSON string
        self.llm = llm
        self.sentence_model = SentenceTransformer(sentence_model_id)
        if torch.cuda.is_available():
            self.sentence_model = self.sentence_model.to(torch.device("cuda"))
            print("QGenerator: SentenceTransformer model moved to GPU.")
        else:
            print("QGenerator: GPU not available for SentenceTransformer, using CPU.")

    # def remove_think_blocks(self, text: str) -> str:
    #     """Removes <think>...</think> blocks from the text."""
    #     if text is None: # None 체크 추가
    #         return ""
    #     return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    def generate(self) -> str:
        prompt = f"도구 목록: {self.tool_schemas_json_str}\\nq를 생성해줘" # Use JSON string
        q = self.llm.generate(prompt)
        return q

    def generate_candidate_qs_with_llm(
        self,
        num_to_generate: int,
        existing_qs_list: List[str],
    ) -> List[str]:
        """Generates candidate 'q' strings using the LLM, removing think blocks."""
        system_prompt = """Your ONLY task is to generate realistic, diverse user queries or requests ('q') suitable for an AI assistant with access to specific tools.
        These queries should be answerable using the provided tools. Vary the complexity, phrasing (questions, commands), and the tools potentially required.

        **CRITICAL INSTRUCTIONS:**
        1.  Output ONLY the raw user queries.
        2.  Each query MUST be on a new line.
        3.  **ABSOLUTELY DO NOT** include:
            * Explanations, comments, or justifications.
            * Numbered lists, bullet points, or any formatting other than one query per line.
            * Any text before the first query or after the last query.
        """

        examples_prompt = ""
        if existing_qs_list:
            sample_existing = random.sample(
                existing_qs_list, min(len(existing_qs_list), 5)
            )
            examples_prompt = (
                "Critically, avoid generating queries too similar to these examples:\n- "
                + "\n- ".join(sample_existing)
                + "\n\n"
            )

        user_prompt = f"""Based on the following available tools: {self.tool_schemas_json_str} 
    Generate exactly {num_to_generate} diverse user queries ('q'). Remember to vary the required tools, complexity, and phrasing. {examples_prompt}Output ONLY the queries, one per line:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            completion, _ = self.llm.chat(
                messages=messages,
                kwargs={
                    "temperature": 0.8, # 변경: 0.7 -> 0.8 (노트북 참고)
                    "max_tokens": 4096,
                },
            )
        except Exception as e:
            print(f"An unexpected error occurred during LLM call: {e}")
            return []

        cleaned_output = completion # LLM client now handles think block removal
        candidate_qs = []
        raw_lines = cleaned_output.split("\n")
        for line in raw_lines:
            clean_line = line.strip()
            if self.__is_valid_query_line(clean_line):
                candidate_qs.append(clean_line)
            elif clean_line:
                print(f"Filtered out invalid line: '{clean_line}'")
        return candidate_qs

    def generate_batch_qs(
        self,
        num_to_generate: int,
        existing_qs_list: List[str],
        llm_batch_size: int = 50,
        # llm_overgen_factor: float = 2.0, # Replaced by iterative attempt logic
        similarity_threshold: float = 0.8,
        top_n_similar: int = 10, # 노트북 참고 파라미터 추가
        max_attempts: int = 5, # Max attempts to reach num_to_generate
    ) -> List[Dict[str, Any]]: # 반환 타입 변경
        """
        LLM에서 대량 쿼리 생성 후, 유사도 필터링을 통해 중복 없이 num_to_generate개 반환.
        각 쿼리에 대한 유사도 정보도 함께 반환.
        Iteratively attempts to generate and filter queries until num_to_generate is met or max_attempts is reached.
        """
        if existing_qs_list is None:
            existing_qs_list = []

        accepted_query_objects = []
        # all_qs_for_compare_strings will start with existing_qs_list and grow with accepted queries
        all_qs_for_compare_strings = list(existing_qs_list)

        all_embeddings_for_compare = None
        if all_qs_for_compare_strings:
            all_embeddings_for_compare = self.sentence_model.encode(all_qs_for_compare_strings, convert_to_tensor=True)

        attempts_count = 0
        print(f"DEBUG: Starting generate_batch_qs. Target: {num_to_generate}, Existing: {len(existing_qs_list)}, Similarity Threshold: {similarity_threshold}") # ADDED
        while len(accepted_query_objects) < num_to_generate and attempts_count < max_attempts:
            attempts_count += 1
            print(f"DEBUG: Attempt {attempts_count}/{max_attempts}. Accepted so far: {len(accepted_query_objects)}/{num_to_generate}") # ADDED
            
            num_still_needed = num_to_generate - len(accepted_query_objects)
            # Determine how many to request from LLM in this attempt
            # Similar to notebook: min(batch_size, needed + buffer)
            num_to_request_this_attempt = min(llm_batch_size, num_still_needed + 5)

            if num_to_request_this_attempt <= 0:
                 print(f"DEBUG: No more queries needed or num_to_request_this_attempt is {num_to_request_this_attempt}. Breaking.") # ADDED
                 break # Should only happen if num_still_needed became <=0 due to concurrent modification or error

            # Generate candidate strings
            # Pass current `all_qs_for_compare_strings` to LLM to avoid generating similar ones to already accepted/existing
            print(f"DEBUG: Attempt {attempts_count}: Requesting {num_to_request_this_attempt} queries from LLM. Total needed: {num_still_needed}.") # MODIFIED (was commented)
            candidate_qs_strings = self.generate_candidate_qs_with_llm(
                num_to_request_this_attempt,
                all_qs_for_compare_strings # Pass the most up-to-date list for LLM's awareness
            )

            if not candidate_qs_strings:
                print(f"DEBUG: LLM generated no candidates in attempt {attempts_count}.") # MODIFIED (was commented)
                if attempts_count == 1 and not accepted_query_objects and not existing_qs_list:
                    print(f"DEBUG: LLM generated no candidates on first attempt with no existing/accepted queries.") # MODIFIED (was commented)
                continue # Try next attempt if any left
            
            print(f"DEBUG: LLM generated {len(candidate_qs_strings)} candidates. First 3: {candidate_qs_strings[:3]}") # ADDED

            # Embed new candidates
            if not candidate_qs_strings: # Should be caught above, but defensive check
                candidate_embeddings = torch.empty(0)
            else:
                candidate_embeddings = self.sentence_model.encode(candidate_qs_strings, convert_to_tensor=True)
            
            if candidate_embeddings.nelement() == 0: # Check if tensor is empty
                print(f"DEBUG: No valid embeddings for candidates in attempt {attempts_count}.") # MODIFIED (was commented)
                continue

            newly_accepted_this_attempt_count = 0
            for idx, q_new in enumerate(candidate_qs_strings):
                if len(accepted_query_objects) >= num_to_generate:
                    break # Overall goal met

                # Ensure idx is within bounds of candidate_embeddings
                if idx >= candidate_embeddings.shape[0]:
                    # This might happen if candidate_qs_strings had empty strings that were filtered out by encode
                    # print(f"Warning: Index {idx} out of bounds for candidate_embeddings (shape {candidate_embeddings.shape[0]})")
                    continue

                current_cand_embedding = candidate_embeddings[idx].unsqueeze(0) # [1, dim]
                q_new_lower = q_new.lower()

                is_exact_duplicate = any(q_new_lower == q_old.lower() for q_old in all_qs_for_compare_strings)
                if is_exact_duplicate:
                    print(f"DEBUG: Query '{q_new}' is an exact duplicate. Skipping.") # ADDED
                    continue

                max_similarity = 0.0
                avg_similarity = 0.0
                similarities_list_for_heap = []
                most_similar_existing_q = "N/A" # ADDED

                if all_embeddings_for_compare is not None and all_embeddings_for_compare.shape[0] > 0:
                    cosine_scores = util.pytorch_cos_sim(current_cand_embedding, all_embeddings_for_compare)[0]
                    cosine_scores_cpu = cosine_scores.cpu().numpy()

                    if cosine_scores_cpu.size > 0:
                        max_similarity = np.max(cosine_scores_cpu)
                        avg_similarity = np.mean(cosine_scores_cpu)
                        # Use a slice of all_qs_for_compare_strings that matches all_embeddings_for_compare
                        # This assumes all_qs_for_compare_strings and all_embeddings_for_compare are kept in sync
                        relevant_comparison_qs = all_qs_for_compare_strings[:all_embeddings_for_compare.shape[0]] # ADDED for safety
                        similarities_list_for_heap = list(zip(cosine_scores_cpu, relevant_comparison_qs)) # MODIFIED
                        if similarities_list_for_heap: 
                            # Find the index of the max score to correctly identify the most similar query
                            max_score_idx = np.argmax(cosine_scores_cpu)
                            if max_score_idx < len(relevant_comparison_qs):
                                most_similar_existing_q = relevant_comparison_qs[max_score_idx]
                
                print(f"DEBUG: Candidate '{q_new}'. Max similarity: {max_similarity:.4f} (Threshold: {similarity_threshold}). Most similar to: '{most_similar_existing_q}'") # ADDED
                
                if max_similarity > similarity_threshold:
                    print(f"DEBUG: Query '{q_new}' rejected due to high similarity.") # ADDED
                    continue

                # Passed filters, accept this query
                most_similar_dict = {}
                if similarities_list_for_heap:
                    actual_top_n = min(top_n_similar, len(similarities_list_for_heap))
                    top_n_items = heapq.nlargest(actual_top_n, similarities_list_for_heap, key=lambda item: item[0])
                    most_similar_dict = {q_text: round(float(score), 4) for score, q_text in top_n_items if float(score) > 0.01}

                new_obj = {
                    "q": q_new,
                    "max_similarity_score_against_all": round(float(max_similarity), 4),
                    "avg_similarity_score": round(float(avg_similarity), 4),
                    "most_similar_instructions": most_similar_dict,
                }
                accepted_query_objects.append(new_obj)
                newly_accepted_this_attempt_count += 1

                # Update comparison basis for the *next* candidate in *this* attempt and for *next* attempt
                all_qs_for_compare_strings.append(q_new)
                if all_embeddings_for_compare is None:
                    all_embeddings_for_compare = current_cand_embedding
                else:
                    all_embeddings_for_compare = torch.cat((all_embeddings_for_compare, current_cand_embedding), dim=0)
            
            if newly_accepted_this_attempt_count == 0 and len(candidate_qs_strings) > 0 : # MODIFIED (was commented)
                print(f"DEBUG: No new diverse queries accepted in attempt {attempts_count} from {len(candidate_qs_strings)} candidates.") # MODIFIED (was commented)
            elif newly_accepted_this_attempt_count > 0:
                print(f"DEBUG: Accepted {newly_accepted_this_attempt_count} queries in attempt {attempts_count}.") # MODIFIED (was commented)

        # End of while loop for attempts
        if len(accepted_query_objects) < num_to_generate: # MODIFIED (was commented)
           print(f"DEBUG: Could only generate {len(accepted_query_objects)} out of {num_to_generate} desired queries after {max_attempts} attempts.") # MODIFIED (was commented)
        else: # ADDED
           print(f"DEBUG: Successfully generated {len(accepted_query_objects)}/{num_to_generate} queries.") # ADDED
        return accepted_query_objects

    def __is_valid_query_line(self, q_text: str) -> bool:
        """Checks if a single line looks like a valid user query."""
        q_text = q_text.strip()
        if not q_text:
            return False
        if q_text.startswith(
            (
                "- ", "* ", "Okay,", "First,", "Next,", "Now,", "Let me", "Wait,", "Also,",
                "###", "//", "```", "Queries", "That's", "This should", "Avoid", "Check for",
                "Example", "User Query:", "Generated Query:",
            )
        ):
            return False
        if q_text.endswith(":") or "→" in q_text:
            return False
        if re.match(r"^\d+\.", q_text):
            return False
        if len(q_text.split()) <= 1 and not re.search(r"[a-zA-Z]", q_text):
            return False
        if "/" in q_text and "." in q_text and " " not in q_text:
            return False
        
        # 노트북의 필터링 로직 강화 (툴 이름 및 관련 키워드)
        example_tool_names = ["get_weather", "get_news", "get_current_location", "create_calendar_event", "fetch_calendar_events", "authorize_calendar_access", "web_search"]
        q_text_lower = q_text.lower()
        if any(tool_name_token in q_text_lower for tool_name_token in example_tool_names):
            if any(keyword in q_text_lower for keyword in ["parameter", "require", "tool", "function", "schema", "api", "endpoint"]):
                return False
        return True

# Helper function to load 'q' strings from a JSON file
def load_qs_strings_from_file(filename: str) -> List[str]:
    qs_list = []
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'q' in item and isinstance(item['q'], str):
                        qs_list.append(item['q'])
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading or parsing {filename}: {e}. Returning empty list.")
            return []
    return qs_list

if __name__ == "__main__":
    # --- MockLLMClient is removed, actual LLMClient will be used. ---

    print("Initializing QGenerator with actual ToolManager and LLMClient...")
    # Use actual LLMClient
    # Ensure FRIENDLI_TOKEN and optionally HF_TOKEN environment variables are set
    try:
        llm_client_instance = LLMClient() 
    except Exception as e:
        print(f"Failed to initialize LLMClient: {e}")
        print("Please ensure your FRIENDLI_TOKEN (and HF_TOKEN if needed for the tokenizer) environment variables are set.")
        exit(1)
        
    # Use actual ToolManager
    tool_manager_instance = ToolManager(llm=llm_client_instance) 
    
    q_generator = QGenerator(tool_manager=tool_manager_instance, llm=llm_client_instance)
    
    print("\\\\n--- Testing generate_batch_qs ---")
    
    QUERIES_FILENAME = "diverse_queries_with_scores_v4.json"
    # Construct the full path to the queries file in the data directory
    queries_file_path = os.path.join("data", QUERIES_FILENAME)
    
    existing_queries_for_test = load_qs_strings_from_file(queries_file_path)
    
    if not existing_queries_for_test:
        print(f"No existing queries loaded from {queries_file_path}, using default test queries.")
        # existing_queries_for_test = ["오늘 서울 날씨 알려줘", "뉴스 찾아줘 IT 분야로", "점심 메뉴 뭐 먹지?"]
        existing_queries_for_test = []
    else:
        print(f"Loaded {len(existing_queries_for_test)} existing queries from {queries_file_path}")

    generated_q_objects = q_generator.generate_batch_qs(
        num_to_generate=20,
        existing_qs_list=existing_queries_for_test,
        llm_batch_size=15,
        # llm_overgen_factor=1.8, # This argument is no longer used
        similarity_threshold=0.85,
        top_n_similar=3
    )

    print(f"\n--- Generated {len(generated_q_objects)} diverse query objects ---")
    for i, q_obj in enumerate(generated_q_objects):
        print(f"Query {i+1}:")
        print(f"  q: {q_obj['q']}") # 수정된 f-string
        print(f"  Max Similarity: {q_obj['max_similarity_score_against_all']:.4f}")
        print(f"  Avg Similarity: {q_obj['avg_similarity_score']:.4f}")
        print(f"  Most Similar Instructions: {q_obj['most_similar_instructions']}")
        if i >= 4: 
            print("  ...")
            break
            
    print("\n--- Testing generate_candidate_qs_with_llm (raw output) ---")
    candidate_strings = q_generator.generate_candidate_qs_with_llm(5, existing_queries_for_test)
    print(f"Generated {len(candidate_strings)} candidate strings (first 5): {candidate_strings[:5]}")
