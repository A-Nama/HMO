# Hear Me Out

**Team:** barbie.dev  

## About the Project

Hear Me Out is a GenAI-powered tool that transforms fleeting voice or text ideas into structured pitch decks. Whether you're a founder in the making, a student, or simply someone who wants to preserve and polish their thoughts, this tool helps you document ideas in the form of ready-to-share presentation decks.

## Features

- Record your idea using voice
- Type out your idea if you prefer text
- Transcription and content structuring using Gemini AI
- Automatically organizes content into a 10-slide pitch deck
- Converts the deck into a downloadable PPT
- Uploads to Supabase storage
- Returns a public link to the deck and stores metadata in a Supabase table

## Tech Stack

**Backend**  
- FastAPI (Python)  
- Google Generative AI (Gemini Pro, Text & Audio)   
- Supabase ( Database and Storage)

**Frontend (Prototype)**  
- Basic HTML + Tailwind CSS  
- Plan to scale with React for a richer interface

**Database Table: `pitch_decks`**  
- Fields: `company_name`, `tagline`, `original_idea`, `pdf_url`, `storage_path`

## How It Works

1. The user records or types out their idea.
2. The input is sent to Gemini AI to transcribe (if audio) and structure the content.
3. A 10-slide pitch deck is generated.
4. The content is formatted into a PDF.
5. The PPT is uploaded to Supabase.
6. A public link is returned, and metadata is stored.

## Scalability and Future Scope

- Multiple pitch deck templates based on context and tone
- Google login and user authentication
- Slide editing before PDF generation
- Richer frontend for mobile and web platforms

## Example Demo Idea

**Startup Name:** FridgePal  
Description: A smart fridge assistant that tracks your groceries, sends expiration alerts, and suggests recipes based on what's available.

The user speaks this idea, and Hear Me Out generates a polished, themed pitch deck with title, problem, solution, market, monetization, and more.

## Installation

```bash
git clone https://github.com/yourusername/hearmeout
cd hearmeout
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```


