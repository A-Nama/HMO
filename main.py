import os
import json
import asyncio
import httpx
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, Header
from pydantic import BaseModel, Field
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv
from supabase import create_client, Client
from fpdf import FPDF
from datetime import datetime
from io import BytesIO
import uuid


# --- Environment Variable Setup ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Pitch Deck Generator API",
    description="An API that uses Google's Gemini to generate and save a pitch deck from a raw idea or audio.",
    version="2.0.0"
)

async def verify_user(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid token format.")
    token = authorization.split(" ")[1]

    try:
        user = supabase.auth.get_user(token)
        if not user.user:
            raise HTTPException(status_code=403, detail="Invalid or expired token.")
        return user.user
    except Exception as e:
        raise HTTPException(status_code=403, detail="Unauthorized.")

# --- CORS Configuration ---
origins = ["http://localhost", "http://localhost:3000", "http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Client Initialization ---
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")
genai.configure(api_key=GEMINI_API_KEY)

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None
    print("FATAL: Supabase credentials not found. Application cannot function without them.")

# --- Pydantic Models ---
class PitchDeckSlide(BaseModel):
    title: str
    content: str
    
class PitchDeckContent(BaseModel):
    company_name: str
    tagline: str
    slides: List[PitchDeckSlide]

# --- AI & PDF Helper Functions (These remain the same) ---

def create_pitch_deck_prompt(idea_text: str) -> str:
    return f"""
    You are an expert startup consultant. Based on the user's idea: "{idea_text}", generate the content for a 10-slide pitch deck.
    Return a single, valid JSON object with the structure: {{"company_name": "...", "tagline": "...", "slides": [{{"title": "...", "content": "..."}}]}}.
    The 10 slides MUST be: 1. The Problem, 2. Our Solution, 3. Product/Service, 4. Market Size, 5. Business Model, 6. Go-to-Market Strategy, 7. Competitive Landscape, 8. The Team, 9. Financial Projections, 10. The Ask.
    """

async def _generate_deck_content(idea_text: str) -> dict:
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = create_pitch_deck_prompt(idea_text)
        response = await model.generate_content_async(prompt)
        response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text content: {e}")

async def _generate_image_for_slide(slide_content: str, client: httpx.AsyncClient) -> bytes:
    if not HUGGING_FACE_API_KEY:
        raise ValueError("HUGGING_FACE_API_KEY not found. Please add it to your .env file.")
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1-base"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
    image_prompt = f"A professional, minimalist, abstract vector art representing the concept of: {slide_content}"
    try:
        response = await client.post(api_url, headers=headers, json={"inputs": image_prompt}, timeout=120.0)
        if response.status_code == 503:
            await asyncio.sleep(15)
            response = await client.post(api_url, headers=headers, json={"inputs": image_prompt}, timeout=120.0)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during deck generation: {str(e)}")
    
async def create_pdf_and_upload(deck: PitchDeckOutput, idea_text: str):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=14)

    # Title slide
    pdf.set_font("Arial", "B", 20)
    pdf.cell(200, 10, txt=deck.company_name, ln=True, align="C")
    pdf.set_font("Arial", "I", 14)
    pdf.cell(200, 10, txt=deck.tagline, ln=True, align="C")
    pdf.ln(10)

    # Other slides
    for slide_key in [
        "problem", "solution", "product", "marketSize", "businessModel",
        "goToMarket", "competition", "team", "financials", "theAsk"
    ]:
        slide = getattr(deck, slide_key)
        pdf.set_font("Arial", "B", 16)
        pdf.multi_cell(0, 10, txt=slide.title)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, txt=slide.content)
        pdf.ln(5)

    # Save to in-memory buffer
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)

    # Generate a unique filename
    filename = f"{deck.company_name}_{uuid.uuid4().hex[:8]}.pdf"
    storage_path = f"{filename}"

    # Upload to Supabase
    res = supabase.storage.from_("pitch_decks").upload(
        path=storage_path,
        file=buffer,
        file_options={"content-type": "application/pdf", "cache-control": "3600"}
    )

    if res.get("error"):
        raise HTTPException(status_code=500, detail=f"Supabase Storage error: {res['error']['message']}")

    # Get public URL
    public_url = supabase.storage.from_("pitch_decks").get_public_url(storage_path)

    return {
        "pdf_url": public_url,
        "storage_path": storage_path
    }


# --- API Endpoints ---

@app.post("/decks", tags=["Pitch Decks"])
async def generate_and_save_deck(audio_file: UploadFile = File(...)):
    """
    Generates a new pitch deck from audio, saves it to the database and
    storage, and returns its metadata.
    """
    if not supabase:
        raise HTTPException(status_code=503, detail="Database connection not available.")
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")

    try:
        # Steps 1-4: Transcribe, Generate Content, Generate Images, Create PDF
        transcription_model = genai.GenerativeModel('gemini-1.5-flash')
        audio_bytes = await audio_file.read()
        uploaded_audio = genai.upload_file(path="temp_audio", content=audio_bytes)
        response = await transcription_model.generate_content_async(["Transcribe this startup idea.", uploaded_audio])
        transcribed_idea = response.text.strip()

        deck_content = await _generate_deck_content(transcribed_idea)

        async with httpx.AsyncClient() as client:
            image_tasks = [_generate_image_for_slide(slide['content'], client) for slide in deck_content['slides']]
            images = await asyncio.gather(*image_tasks)

        pdf_buffer = _create_pitch_deck_pdf(deck_content, images)
        pdf_bytes = pdf_buffer.getvalue()
        
        # Step 5: Upload PDF to Supabase Storage
        file_path = f"public/{uuid.uuid4()}.pdf"
        # The bucket name is 'pitch_decks'
        supabase.storage.from_("pitch_decks").upload(file=pdf_bytes, path=file_path, file_options={"content-type": "application/pdf"})

        # Step 6: Get Public URL for the PDF
        res = supabase.storage.from_("pitch_decks").get_public_url(file_path)
        pdf_url = res

        # Step 7: Save Metadata to Supabase Database
        deck_to_save = {
            "company_name": deck_content.get("company_name"),
            "tagline": deck_content.get("tagline"),
            "original_idea": transcribed_idea,
            "pdf_url": pdf_url,
            "storage_path": file_path
        }
        db_response = supabase.table("pitch_decks").insert(deck_to_save).execute()
        
        # Step 8: Return the newly created record
        new_deck = db_response.data[0]
        return JSONResponse(status_code=201, content=new_deck)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
    
@app.post("/generate-from-text", response_model=PitchDeckOutput, tags=["Pitch Deck Generation"])
async def generate_from_text(payload: IdeaInput):
    """
    Accepts an idea in plain text and generates a pitch deck.
    """
    try:
        deck_data = await _generate_deck_from_text(payload.idea)
        deck_data['original_idea'] = payload.idea
        return deck_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/save-deck", tags=["Database"])
async def save_deck(payload: SaveDeckInput, user=Depends(verify_user)):
    """
    Accepts a generated pitch deck and saves it to the Supabase database,
    including uploading the PDF to Supabase Storage.
    """
    try:
        # Upload the PDF and get its URL and path
        upload_info = await create_pdf_and_upload(payload.deck_data, payload.original_idea)

        data_to_insert = {
            "original_idea": payload.original_idea,
            "company_name": payload.deck_data.company_name,
            "tagline": payload.deck_data.tagline,
            "pdf_url": upload_info["pdf_url"],
            "storage_path": upload_info["storage_path"],
        }

        # Insert into Supabase
        data, count = supabase.table('pitch_decks').insert(data_to_insert).execute()
        response_data = data[1][0] if data and len(data[1]) > 0 else None

        if not response_data:
            raise HTTPException(status_code=500, detail="Failed to save data to Supabase.")

        return {
            "status": "success",
            "message": "Deck saved and PDF uploaded successfully!",
            "saved_id": response_data.get("id"),
            "pdf_url": upload_info["pdf_url"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
