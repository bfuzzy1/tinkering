from mirascope.core import openai
from pydantic import BaseModel, Field
from typing import List
import lilypad

lilypad.configure()

class OutlineSection(BaseModel):
    title: str = Field(..., description="Section title")
    key_points: List[str] = Field(..., description="Key points to cover in this section")
    subsections: List['OutlineSection'] = Field(default=[], description="Optional nested subsections")

class OutlineCriteria(BaseModel):
    meets_requirements: bool = Field(..., description="Whether the outline meets all criteria")
    missing_elements: List[str] = Field(default=[], description="List of missing required elements")
    suggestions: List[str] = Field(default=[], description="Suggestions for improvement")

class DocumentContent(BaseModel):
    content: str = Field(..., description="The generated document content")
    sections_covered: List[str] = Field(..., description="List of sections covered in the document")

class DocumentWorkflowAgent(BaseModel):
    messages: list = []
    max_retries: int = Field(default=3, description="Maximum number of outline revision attempts")

    @lilypad.generation()
    @openai.call("o3-mini", response_model=List[OutlineSection])
    async def create_outline(self, topic: str, requirements: str, feedback: str = "") -> str:
        """Create a document outline based on the topic and requirements."""
        prompt = f"""Create a structured document outline for: {topic}
                    Requirements: {requirements}
                    """
        if feedback:
            prompt += f"\nPrevious feedback to address: {feedback}"
        return prompt

    @lilypad.generation()
    @openai.call("o3-mini", response_model=OutlineCriteria)
    async def validate_outline(self, outline: List[OutlineSection], criteria: str) -> str:
        """Validate if the outline meets specified criteria."""
        return f"""Evaluate this outline against the following criteria:
                  {criteria}
                  
                  Outline to evaluate:
                  {outline}  
                """

    @lilypad.generation()
    @openai.call("o3-mini", response_model=DocumentContent)
    async def generate_document(self, outline: List[OutlineSection], validation: OutlineCriteria = None) -> str:
        """Generate the full document based on the approved outline."""
        prompt = f"""Write a complete document following this outline:
                  {outline}
                  
                  Ensure each section addresses all key points listed."""
        
        if validation and validation.suggestions:
            prompt += "\nPlease incorporate these suggestions:\n"
            for suggestion in validation.suggestions:
                prompt += f"- {suggestion}"
        
        return prompt

    async def run(self, topic: str, requirements: str, criteria: str):
        """Execute the full document creation workflow."""
        try:
            attempt = 0
            outline = None
            validation = None
            
            while attempt < self.max_retries:
                # Step 1: Create/revise outline
                print(f"\nAttempt {attempt + 1}: {'Creating' if attempt == 0 else 'Revising'} outline...")
                feedback = f"{validation.missing_elements}\n{validation.suggestions}" if validation else ""
                outline = await self.create_outline(topic, requirements, feedback)
                print(f"{'Initial' if attempt == 0 else 'Revised'} outline created: {outline}")

                # Step 2: Validate outline
                print("Validating outline...")
                validation = await self.validate_outline(outline, criteria)
                print(f"Validation results: {validation}")

                # Check if requirements are met
                if validation.meets_requirements:
                    print("\nOutline meets all requirements!")
                    break
                
                attempt += 1
                if attempt == self.max_retries:
                    print(f"\nFailed to create satisfactory outline after {self.max_retries} attempts")
                    return f"Failed to create satisfactory outline. Last validation: {validation}"

            # Step 3: Generate document if outline is approved
            print("\nGenerating final document...")
            document = await self.generate_document(outline, validation)
            return document.content

        except Exception as e:
            print(f"Error in workflow: {str(e)}")
            raise

if __name__ == "__main__":
    import asyncio

    async def main():
        agent = DocumentWorkflowAgent()
        topic = "The Impact of Artificial Intelligence on Cybersecurity"
        requirements = """
        - Must include at least 5-6 main sections
        - Each section should have 5-6 key points
        - Must cover both benefits and challenges
        - Should include real-world examples
        - We are writing for an CISO audience so this must be technical and detailed.
        """
        criteria = """
        - All required sections present
        - Balanced coverage of topics
        - Logical flow between sections
        - Sufficient detail in key points
        """
        
        result = await agent.run(topic, requirements, criteria)
        print(result)

    asyncio.run(main())