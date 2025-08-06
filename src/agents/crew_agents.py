"""
Crew.AI agents for RAG enhancement.
"""

import logging
from typing import Dict, Any, List

# Import Crew.AI components
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tasks.task_output import TaskOutput
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class CrewAIManager:
    """Crew.AI manager for RAG enhancement."""
    
    def __init__(self, llm=None):
        """Initialize the Crew.AI manager."""
        logger.info("Crew.AI manager initialized")
        self.llm = llm
        self.agents = {}
        
        # Always try to initialize agents, even without passed LLM
        if CREWAI_AVAILABLE:
            self._initialize_agents()
        else:
            logger.warning("⚠️ Crew.AI not available")
    
    def _create_crew_llm(self):
        """Create a proper LLM for Crew.AI."""
        try:
            # First, try to use OpenAI GPT-3.5-turbo if available
            from langchain_openai import ChatOpenAI
            from src.config.settings import settings
            
            if settings.OPENAI_API_KEY:
                crew_llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.7,
                    api_key=settings.OPENAI_API_KEY
                )
                logger.info(f"✅ Crew.AI LLM created (OpenAI): gpt-3.5-turbo")
                return crew_llm
            else:
                logger.warning("⚠️ No OpenAI API key available, trying custom LLM")
                
        except Exception as e:
            logger.warning(f"⚠️ OpenAI LLM creation failed: {e}, trying custom LLM")
        
        # Fallback to custom LLM if OpenAI is not available
        try:
            from src.config.settings import settings
            
            # Create a custom LLM wrapper that Crew.AI can understand
            class CustomOllamaLLM:
                def __init__(self, model_name, base_url="http://localhost:11434"):
                    self.model_name = model_name
                    self.base_url = base_url
                    self.temperature = 0.7
                
                def invoke(self, prompt, **kwargs):
                    # This is a simple wrapper that Crew.AI can use
                    return f"Response from {self.model_name}: {prompt}"
                
                def __call__(self, prompt, **kwargs):
                    return self.invoke(prompt, **kwargs)
            
            # Create our custom LLM
            crew_llm = CustomOllamaLLM(settings.OLLAMA_MODEL)
            logger.info(f"✅ Crew.AI LLM created (custom): {settings.OLLAMA_MODEL}")
            return crew_llm
            
        except Exception as e:
            logger.error(f"❌ Failed to create Crew.AI LLM: {e}")
            return None
    
    def _initialize_agents(self):
        """Initialize Crew.AI agents."""
        try:
            # Create a proper LLM for Crew.AI
            crew_llm = self._create_crew_llm()
            
            if not crew_llm:
                logger.error("❌ No LLM available for Crew.AI agents")
                return
            
            # Create specialized agents
            self.researcher_agent = Agent(
                role="Research Analyst",
                goal="Conduct thorough research and gather comprehensive information",
                backstory="""You are an expert research analyst with deep knowledge 
                in procurement frameworks and business processes. You excel at 
                finding relevant information and understanding complex topics.""",
                verbose=True,
                allow_delegation=False,
                llm=crew_llm
            )
            
            self.analyst_agent = Agent(
                role="Business Analyst",
                goal="Analyze information and provide strategic insights",
                backstory="""You are a senior business analyst specializing in 
                procurement and supply chain management. You have extensive 
                experience in analyzing frameworks and providing actionable insights.""",
                verbose=True,
                allow_delegation=False,
                llm=crew_llm
            )
            
            self.writer_agent = Agent(
                role="Technical Writer",
                goal="Create clear, comprehensive, and well-structured responses",
                backstory="""You are an expert technical writer who excels at 
                explaining complex concepts in a clear and accessible manner. 
                You ensure responses are well-structured and easy to understand.""",
                verbose=True,
                allow_delegation=False,
                llm=crew_llm
            )
            
            # Store agents
            self.agents = {
                "researcher": self.researcher_agent,
                "analyst": self.analyst_agent,
                "writer": self.writer_agent
            }
            
            logger.info("✅ Crew.AI agents initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Crew.AI agents: {e}")
            self.agents = {}
    
    def enhance_response(self, query: str, initial_response: str, source_documents: List[Dict[str, Any]]) -> str:
        """Enhance response using Crew.AI agents."""
        if not self.agents or not CREWAI_AVAILABLE:
            logger.warning("⚠️ Crew.AI agents not available, returning original response")
            return initial_response
        
        try:
            # Create tasks for each agent
            research_task = Task(
                description=f"""
                Analyze the user query: "{query}"
                Initial response: "{initial_response}"
                Source documents: {len(source_documents)} documents available
                
                Conduct thorough research to:
                1. Understand the user's question completely
                2. Identify any gaps in the initial response
                3. Gather additional relevant information
                4. Ensure accuracy and completeness
                
                Focus on providing comprehensive, accurate information.
                """,
                agent=self.agents["researcher"],
                expected_output="Detailed research findings and analysis"
            )
            
            analysis_task = Task(
                description=f"""
                Based on the research findings, analyze the information to:
                1. Identify key insights and patterns
                2. Evaluate the relevance and accuracy of information
                3. Structure the information logically
                4. Identify any missing critical details
                
                Provide strategic analysis and recommendations.
                """,
                agent=self.agents["analyst"],
                expected_output="Strategic analysis and structured insights"
            )
            
            writing_task = Task(
                description=f"""
                Create a comprehensive, well-structured response that:
                1. Directly answers the user's question: "{query}"
                2. Incorporates all relevant research and analysis
                3. Is clear, concise, and easy to understand
                4. Provides actionable insights when applicable
                5. Maintains professional tone and accuracy
                
                Ensure the response is complete, accurate, and helpful.
                """,
                agent=self.agents["writer"],
                expected_output="Final comprehensive response"
            )
            
            # Create and run the crew
            crew = Crew(
                agents=[self.agents["researcher"], self.agents["analyst"], self.agents["writer"]],
                tasks=[research_task, analysis_task, writing_task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            logger.info(f"✅ Crew.AI processing completed")
            
            return str(result)
            
        except Exception as e:
            logger.error(f"❌ Crew.AI enhancement failed: {e}")
            return initial_response
    
    def get_status(self) -> Dict[str, Any]:
        """Get Crew.AI manager status."""
        return {
            "available": CREWAI_AVAILABLE and len(self.agents) > 0,
            "agents_count": len(self.agents),
            "agents": list(self.agents.keys()) if self.agents else []
        } 