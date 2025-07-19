import os
import json
from fastapi import FastAPI, HTTPException, Body, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv
from supabase import create_client, Client

# --- Environment Variable Setup ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Pitch Deck Generator API",
    description="An API that uses Google's Gemini to generate and save a pitch deck from a raw idea or audio.",
    version="2.0.0"
)

# --- CORS Configuration ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    # Add your deployed frontend URL here
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Client Initialization ---
# Gemini
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")
genai.configure(api_key=GEMINI_API_KEY)

# Supabase
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL or SUPABASE_KEY not found. Please set them in your .env file.")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# --- Pydantic Models ---
class IdeaInput(BaseModel):
    idea: str = Field(..., min_length=20, description="The user's raw idea for a startup/project.")

class PitchDeckSlide(BaseModel):
    title: str
    content: str
    
class PitchDeckOutput(BaseModel):
    company_name: str
    tagline: str
    problem: PitchDeckSlide
    solution: PitchDeckSlide
    product: PitchDeckSlide
    marketSize: PitchDeckSlide
    businessModel: PitchDeckSlide
    goToMarket: PitchDeckSlide
    competition: PitchDeckSlide
    team: PitchDeckSlide
    financials: PitchDeckSlide
    theAsk: PitchDeckSlide

class SaveDeckInput(BaseModel):
    """Model for receiving the data to be saved to Supabase."""
    original_idea: str
    deck_data: PitchDeckOutput


# --- AI Prompt Engineering ---
def create_pitch_deck_prompt(idea_text: str) -> str:
    # This function remains the same as before
    return f"""
    You are an expert startup consultant named 'IdeaSpark'. Your task is to generate a comprehensive 10-slide pitch deck based on the user's raw idea.
    The user's idea is: "{idea_text}"
    You MUST return the output as a single, valid JSON object. Do not include any text, markdown formatting, or code fences before or after the JSON object.
    The JSON object must strictly follow this structure:
    {{
      "company_name": "A creative and relevant name for the startup.", "tagline": "A short, powerful tagline.",
      "problem": {{ "title": "The Problem", "content": "A compelling description of the problem." }},
      "solution": {{ "title": "Our Solution", "content": "Clearly explain how you solve the problem." }},
      "product": {{ "title": "Product / Service", "content": "Describe the product and its key features." }},
      "marketSize": {{ "title": "Market Size", "content": "Estimate the market size (TAM, SAM, SOM)." }},
      "businessModel": {{ "title": "Business Model", "content": "How will the company make money?" }},
      "goToMarket": {{ "title": "Go-to-Market Strategy", "content": "How will you reach your target customers?" }},
      "competition": {{ "title": "Competitive Landscape", "content": "Who are the main competitors and what is your advantage?" }},
      "team": {{ "title": "The Team", "content": "Create plausible founder archetypes." }},
      "financials": {{ "title": "Financial Projections", "content": "Provide a high-level 3-year projection." }},
      "theAsk": {{ "title": "The Ask", "content": "State how much funding is being requested and for what." }}
    }}
    """

# --- Internal Helper Functions ---
async def _generate_deck_from_text(idea_text: str) -> dict:
    """Internal logic to generate a pitch deck from text."""
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = create_pitch_deck_prompt(idea_text)
        response = model.generate_content(prompt)
        response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(response_text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI model returned invalid JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during deck generation: {str(e)}")


# --- API Endpoints ---

@app.post("/generate-from-audio", response_model=PitchDeckOutput, tags=["Pitch Deck Generation"])
async def generate_from_audio(audio_file: UploadFile = File(...)):
    """
    Accepts an audio file, transcribes it, and generates a pitch deck.
    This is the primary endpoint your frontend should use for audio input.
    """
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")
        
    try:
        # 1. Transcribe the audio using Gemini
        # The 'gemini-1.5-flash' model is great for this kind of task
        transcription_model = genai.GenerativeModel('gemini-1.5-flash')
        audio_bytes = await audio_file.read()
        
        # Upload the audio file to Google's servers for processing
        uploaded_audio = genai.upload_file(
            path="temp_audio_file", # Temporary name, path doesn't matter with content
            display_name=audio_file.filename,
            content=audio_bytes
        )

        prompt = "Transcribe this audio. It contains a startup idea. Capture the core concept clearly and concisely."
        response = transcription_model.generate_content([prompt, uploaded_audio])
        
        if not response.text:
             raise HTTPException(status_code=500, detail="Transcription failed. The model returned no text.")

        transcribed_idea = response.text.strip()
        
        # 2. Generate the pitch deck from the transcribed text
        deck_data = await _generate_deck_from_text(transcribed_idea)
        
        # Add the transcribed text to the response for the frontend to use
        deck_data['original_idea'] = transcribed_idea
        
        return deck_data

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")


@app.post("/save-deck", tags=["Database"])
async def save_deck(payload: SaveDeckInput):
    """
    Accepts a generated pitch deck and saves it to the Supabase database.
    """
    try:
        data_to_insert = {
            "original_idea": payload.original_idea,
            "deck_data": payload.deck_data.model_dump() 
        }
        
        # Insert data into the 'pitch_decks' table
        data, count = supabase.table('pitch_decks').insert(data_to_insert).execute()
        
        # The response from execute() is a tuple (data, count)
        response_data = data[1][0] if data and len(data[1]) > 0 else None
        
        if not response_data:
            raise HTTPException(status_code=500, detail="Failed to save data to Supabase.")

        return {"status": "success", "message": "Deck saved successfully!", "saved_id": response_data.get('id')}

    except Exception as e:
        print(f"Supabase save error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save to Supabase: {str(e)}")


@app.get("/", tags=["Health Check"])
async def read_root():
    return {"status": "ok", "message": "Welcome to the Pitch Deck Generator API v2!"}