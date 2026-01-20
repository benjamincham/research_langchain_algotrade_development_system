from typing import Dict, Any, List, Optional
from src.core.base_agent import BaseAgent
from src.core.logging import logger
import json

class ValidationAgent(BaseAgent):
    """
    Agent responsible for validating the output of tool calls and research findings.
    
    Ensures data quality, consistency, and adherence to expected formats.
    """
    
    def __init__(
        self,
        name: str = "DataValidator",
        llm: Optional[Any] = None
    ):
        system_prompt = """
        You are the Data Validation Agent. Your primary role is to assess the quality,
        accuracy, and consistency of raw data retrieved by tools or findings generated
        by other agents.
        
        Your responsibilities:
        1. Check if the data adheres to the expected structure (e.g., JSON schema).
        2. Identify any missing or suspicious values (e.g., zero volume, negative price).
        3. Compare multiple data points for consistency (e.g., does the news sentiment match the price action).
        4. Output a clear validation report.
        """
        super().__init__(name=name, role="Data Validator", llm=llm, system_prompt=system_prompt)

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the validation workflow.
        
        Args:
            input_data: Dictionary containing 'data_to_validate' (the raw data/finding) and 'context' (e.g., expected schema, source).
            
        Returns:
            A dictionary containing the validation result and a confidence score.
        """
        data_to_validate = input_data.get("data_to_validate")
        context = input_data.get("context", {})
        
        if not data_to_validate:
            return {
                "validation_status": "FAILED",
                "confidence_score": 0.0,
                "report": "No data provided for validation."
            }
            
        logger.info(f"Starting validation for data from source: {context.get('source', 'Unknown')}")
        
        validation_prompt = self._generate_validation_prompt(data_to_validate, context)
        raw_validation_response = await self._call_llm(validation_prompt)
        
        return self._parse_validation_response(raw_validation_response)

    def _generate_validation_prompt(self, data_to_validate: Any, context: Dict[str, Any]) -> str:
        """Generates the prompt for the LLM to perform validation."""
        
        # Convert data to string for LLM input
        data_str = json.dumps(data_to_validate, indent=2) if isinstance(data_to_validate, (dict, list)) else str(data_to_validate)
        context_str = json.dumps(context, indent=2)
        
        return f"""
        You are the Data Validation Agent. Analyze the following data based on the provided context.
        
        **Context/Expectations:**
        {context_str}
        
        **Data to Validate:**
        ```json
        {data_str}
        ```
        
        Your analysis should focus on:
        1.  **Format Check:** Does the data look like valid {context.get('expected_format', 'JSON')}?
        2.  **Consistency Check:** Are there any values that seem inconsistent or suspicious (e.g., zero volume, negative price, or missing key fields)?
        3.  **Quality Check:** Is the data useful and complete for the intended purpose?
        
        Your output MUST be a JSON object with the following structure:
        {{
            "validation_status": "PASSED" or "FAILED" or "WARNING",
            "confidence_score": "A float between 0.0 and 1.0 representing your confidence in the data's quality.",
            "report": "A concise summary of the validation findings, including any warnings or reasons for failure."
        }}
        
        The final output MUST be only the JSON object.
        """

    def _parse_validation_response(self, raw_response: str) -> Dict[str, Any]:
        """Parses the LLM's validation response."""
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse validation response JSON: {e}. Raw: {raw_response}")
            return {
                "validation_status": "ERROR",
                "confidence_score": 0.0,
                "report": f"Validation Agent failed to produce a valid JSON report. Raw response: {raw_response[:100]}..."
            }
