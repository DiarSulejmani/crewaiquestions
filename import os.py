import os
import json
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool, RAGSearchTool
from langchain_openai import ChatOpenAI

# Initialize OpenAI LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Configuration
class CrewConfig:
    def __init__(self, vector_db_connection_string: str = None):
        self.vector_db_connection_string = vector_db_connection_string
        self.similarity_search_k = 5
        self.embeddings_model = "text-embedding-ada-002"

# Initialize configuration with your PostgreSQL connection
config = CrewConfig(vector_db_connection_string="postgresql+psycopg2://postgres:changeit@db.jlqpiruljvmkvumnbtqd.supabase.co:5432/postgres")

# Initialize RAG tool with your Supabase PostgreSQL database
rag_tool = RAGSearchTool(
    vector_db_path=config.vector_db_connection_string,
    embeddings_model=config.embeddings_model
)

# Vector Database Tools
@tool("vector_database_reader")
def read_from_vector_db(query: str) -> str:
    """Query the vector database for relevant document content."""
    try:
        results = rag_tool.search(query, k=config.similarity_search_k)
        if isinstance(results, list):
            content = "\n\n".join([str(doc) for doc in results])
        else:
            content = str(results)
        return content
    except Exception as e:
        return f"Error reading from vector database: {str(e)}"

# Define Agents
document_reader = Agent(
    role="Document Reader",
    goal="Extract relevant information from PDF documents stored in the vector database",
    backstory="You are an expert document analyst who extracts key information from documents based on specific queries.",
    tools=[read_from_vector_db],
    llm=llm,
    verbose=False
)

question_generator = Agent(
    role="Question Generator",
    goal="Generate questions in JSON format based on document content",
    backstory="""You create questions in two types:
    1. Multiple Choice: 4 options, question_type: "multiple_choice"
    2. Text: Open-ended, question_type: "text", options: null
    
    Output each question as valid JSON with structure:
    {
        "question_text": "Question here",
        "question_type": "multiple_choice" or "text",
        "options": ["A", "B", "C", "D"] or null,
        "correct_answer": "Answer here"
    }""",
    llm=llm,
    verbose=False
)

question_reviewer = Agent(
    role="Question Reviewer",
    goal="Review and output final JSON array of questions",
    backstory="""You review questions and output the final result as a valid JSON array.
    Ensure proper JSON formatting and structure. Output format:
    [
        {
            "question_text": "Question 1",
            "question_type": "multiple_choice",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A"
        },
        {
            "question_text": "Question 2", 
            "question_type": "text",
            "options": null,
            "correct_answer": "Answer"
        }
    ]""",
    llm=llm,
    verbose=False
)

# Define Tasks with distributed parameters
def create_dynamic_tasks(topic: str, keywords: str, num_multiple_choice: int, num_text: int, schwierigkeitsgrad: str, zielgruppe: str, fragetyp: str):
    
    task1 = Task(
        description=f"""Extrahiere relevante Informationen aus der Vektor-Datenbank.
        
        Thema: {topic}
        Keywords: {keywords}
        
        Suche nach Dokumenten die sowohl das Thema "{topic}" als auch die Keywords "{keywords}" behandeln.
        Fokussiere dich auf Inhalte, die für die Erstellung von {fragetyp} geeignet sind.
        
        Erstelle eine strukturierte Zusammenfassung der relevanten Inhalte.""",
        agent=document_reader,
        expected_output="Strukturierte Zusammenfassung relevanter Dokumentinhalte zu Thema und Keywords"
    )
    
    # Difficulty level instructions for Generator and Reviewer
    difficulty_instructions = {
        "leicht": "Erstelle einfache Fragen für Grundlagen und basic Verständnis",
        "mittel": "Erstelle Fragen mit mittlerem Schwierigkeitsgrad für gutes Verständnis", 
        "schwer": "Erstelle anspruchsvolle Fragen für tiefes Verständnis und Analyse"
    }
    
    # Target group instructions for Generator and Reviewer
    target_group_instructions = {
        "bachelor": "Fragen auf Bachelor-Niveau mit grundlegenden bis mittleren Konzepten",
        "master": "Fragen auf Master-Niveau mit fortgeschrittenen Konzepten",
        "abitur": "Fragen auf Abitur-Niveau mit schulischen Konzepten",
        "berufsschule": "Fragen auf Berufsschul-Niveau mit praktischen Anwendungen"
    }
    
    # Question type specific instructions
    fragetyp_instructions = {
        "Rechenfragen": "Erstelle Fragen die Berechnungen, mathematische Formeln oder numerische Probleme beinhalten. Fragen sollen konkrete Rechenaufgaben oder Anwendung von Formeln erfordern.",
        "Verständnisfragen": "Erstelle Fragen die Konzepte, Definitionen, Zusammenhänge und theoretisches Verständnis testen. Fragen sollen Wissen und Verständnis ohne komplexe Berechnungen prüfen."
    }
    
    difficulty_instruction = difficulty_instructions.get(schwierigkeitsgrad, difficulty_instructions["mittel"])
    target_instruction = target_group_instructions.get(zielgruppe, target_group_instructions["bachelor"])
    fragetyp_instruction = fragetyp_instructions.get(fragetyp, fragetyp_instructions["Verständnisfragen"])
    
    task2 = Task(
        description=f"""Generiere {num_multiple_choice} Multiple-Choice und {num_text} Textfragen basierend auf den Dokumentinhalten.
        
        Parameter für Fragenerstellung:
        - Schwierigkeitsgrad: {schwierigkeitsgrad} - {difficulty_instruction}
        - Zielgruppe: {zielgruppe} - {target_instruction}  
        - Fragetyp: {fragetyp} - {fragetyp_instruction}
        
        Für Multiple-Choice Fragen:
        - Genau 4 Antwortoptionen
        - Eine korrekte Antwort
        - question_type: "multiple_choice"
        
        Für Textfragen:
        - Offene Fragen die detaillierte Antworten erfordern
        - options: null
        - question_type: "text"
        
        JSON Format für jede Frage:
        {{
            "question_text": "Frage hier",
            "question_type": "multiple_choice" oder "text",
            "options": ["A", "B", "C", "D"] oder null,
            "correct_answer": "Antwort hier"
        }}""",
        agent=question_generator,
        expected_output="Fragen im JSON Format entsprechend den spezifizierten Parametern",
        context=[task1]
    )
    
    task3 = Task(
        description=f"""Überprüfe und erstelle das finale JSON Array aller Fragen.
        
        Qualitätskriterien basierend auf den Parametern:
        - Schwierigkeitsgrad: {schwierigkeitsgrad} - {difficulty_instruction}
        - Zielgruppe: {zielgruppe} - {target_instruction}
        - Fragetyp: {fragetyp} - {fragetyp_instruction}
        
        Stelle sicher:
        - Korrekte JSON Struktur
        - Angemessene Fragen für Schwierigkeitsgrad und Zielgruppe
        - Fragetyp entspricht den Anforderungen
        - Alle Antworten sind korrekt und vollständig
        
        Ausgabe als valides JSON Array: [{{...}}, {{...}}]""",
        agent=question_reviewer,
        expected_output="Valides JSON Array von qualitätsgeprüften Fragen",
        context=[task1, task2]
    )
    
    return [task1, task2, task3]

# Create Crew
def create_crew():
    return Crew(
        agents=[document_reader, question_generator, question_reviewer],
        tasks=[],
        process=Process.sequential,
        verbose=False
    )

# Main function with dynamic inputs
def generate_questions_json(request_payload: dict, connection_string: str = None):
    """
    Generate questions based on dynamic request payload.
    
    Args:
        request_payload (dict): Input parameters from request
        connection_string (str): Vector database connection string
    
    Expected payload structure:
    {
        "schwierigkeitsgrad": "mittel",     # difficulty level
        "anzahl_fragen": 2,                 # number of questions
        "thema": "Statistik",               # topic/theme
        "fragetyp": "Verständnisfragen",    # Rechenfragen or Verständnisfragen
        "keywords": "rechnen",              # keywords for search
        "zielgruppe": "bachelor"            # target group
    }
    """
    
    # Extract parameters from payload
    schwierigkeitsgrad = request_payload.get("schwierigkeitsgrad", "mittel")
    anzahl_fragen = request_payload.get("anzahl_fragen", 5)
    thema = request_payload.get("thema", "Allgemein")
    fragetyp = request_payload.get("fragetyp", "Verständnisfragen")
    keywords = request_payload.get("keywords", "")
    zielgruppe = request_payload.get("zielgruppe", "bachelor")
    
    # Validate fragetyp
    if fragetyp not in ["Rechenfragen", "Verständnisfragen"]:
        raise ValueError("fragetyp must be either 'Rechenfragen' or 'Verständnisfragen'")
    
    # Map question types to format distribution
    if fragetyp == "Rechenfragen":
        # Rechenfragen are typically text-based for calculations
        num_multiple_choice = 0
        num_text = anzahl_fragen
    else:  # Verständnisfragen
        # Mixed format for understanding questions
        num_multiple_choice = int(anzahl_fragen * 0.6)
        num_text = anzahl_fragen - num_multiple_choice
    
    # Update connection if provided
    if connection_string:
        config.vector_db_connection_string = connection_string
        global rag_tool
        rag_tool = RAGSearchTool(
            vector_db_path=connection_string,
            embeddings_model=config.embeddings_model
        )
    
    # Create and run crew with distributed parameters
    tasks = create_dynamic_tasks(
        # Document Reader gets: thema + keywords
        topic=thema,
        keywords=keywords,
        # Question Generator and Reviewer get: rest of parameters
        num_multiple_choice=num_multiple_choice,
        num_text=num_text,
        schwierigkeitsgrad=schwierigkeitsgrad,
        zielgruppe=zielgruppe,
        fragetyp=fragetyp
    )
    crew = create_crew()
    crew.tasks = tasks
    
    result = crew.kickoff()
    return result

# Usage with your Supabase PostgreSQL database
if __name__ == "__main__":
    # Your Supabase PostgreSQL connection string is already configured above
    CONNECTION_STRING = "postgresql+psycopg2://postgres:changeit@db.jlqpiruljvmkvumnbtqd.supabase.co:5432/postgres"
    
    # Example request payload for Verständnisfragen
    request_payload_verstaendnis = {
        "schwierigkeitsgrad": "mittel",
        "anzahl_fragen": 2,
        "thema": "Statistik", 
        "fragetyp": "Verständnisfragen",
        "keywords": "rechnen",
        "zielgruppe": "bachelor"
    }
    
    # Example request payload for Rechenfragen  
    request_payload_rechnen = {
        "schwierigkeitsgrad": "schwer",
        "anzahl_fragen": 3,
        "thema": "Mathematik",
        "fragetyp": "Rechenfragen", 
        "keywords": "integration ableitung",
        "zielgruppe": "master"
    }
    
    print("Generating Verständnisfragen...")
    result1 = generate_questions_json(
        request_payload=request_payload_verstaendnis,
        connection_string=CONNECTION_STRING
    )
    print(result1)
    
    print("\n" + "="*50 + "\n")
    
    print("Generating Rechenfragen...")
    result2 = generate_questions_json(
        request_payload=request_payload_rechnen,
        connection_string=CONNECTION_STRING
    )
    print(result2)