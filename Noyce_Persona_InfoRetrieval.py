import os
import pandas as pd
from openai import OpenAI
import re
import textwrap
import time

# ############################################################
# # SETUP - Initialize OpenAI client with API key
# ############################################################
client = OpenAI(
    api_key="PUT YOUR API KEY IN HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
)

class PersonaChatbot:
    def __init__(self, personas_file_path, activity_map_file_path):
        """Initialize the chatbot with data files."""
        # ############################################################
        # # DATA LOADING - Read CSV files and create dictionaries
        # ############################################################
        # Load CSV data
        self.personas_df = pd.read_csv(personas_file_path)
        self.activity_map_df = pd.read_csv(activity_map_file_path)
        
        # Create a dictionary of personas for quick access
        self.personas = {}
        for _, row in self.personas_df.iterrows():
            self.personas[row['Name'].lower()] = row.to_dict()
        
        # Create a dictionary of activities for quick access
        self.activities = {}
        for _, row in self.activity_map_df.iterrows():
            self.activities[row['Resident'].lower()] = row.to_dict()
            
        # Store all names for entity recognition
        self.all_names = [name.lower() for name in self.personas.keys()]
    
    def identify_persona(self, query):
        """Identify which persona the query is about."""
        # ############################################################
        # # ENTITY RECOGNITION - Find which resident is mentioned
        # ############################################################
        query_lower = query.lower()
        for name in self.all_names:
            if name in query_lower:
                return name
        return None
    
    def get_persona_info(self, persona_name):
        """Get all information about a specific persona."""
        # ############################################################
        # # DATA RETRIEVAL - Get all data for the identified resident
        # ############################################################
        if persona_name and persona_name in self.personas:
            persona_data = self.personas[persona_name]
            activity_data = self.activities.get(persona_name, {})
            
            # Combine both datasets
            combined_data = {**persona_data, **activity_data}
            return combined_data
        return None
    
    def extract_relevant_info(self, query, persona_data):
        """Extract relevant information from persona data based on the query."""
        # ############################################################
        # # QUERY ANALYSIS - Determine what information is being requested
        # ############################################################
        query_lower = query.lower()
        
        # Define keywords to map to specific data points
        keyword_mapping = {
            "age": ["age", "old", "year"],
            "dementia": ["dementia", "cognitive", "memory", "condition"],
            "mobility": ["mobility", "walk", "wheelchair", "move"],
            "health": ["health", "condition", "medical", "disease", "illness"],
            "hearing": ["hearing", "hear", "deaf", "ear"],
            "eyesight": ["eyesight", "vision", "see", "blind", "glasses"],
            "medication": ["medication", "medicine", "pill", "drug"],
            "sleep": ["sleep", "rest", "nap", "bed", "wake"],
            "personality": ["personality", "character", "behavior", "like", "person"],
            "conversation": ["talk", "speak", "conversation", "communicat", "verbal"],
            "behaviors": ["behavior", "watch", "monitor", "concern"],
            "happy": ["happy", "joy", "like", "enjoy", "love", "favorite"],
            "upset": ["upset", "sad", "angry", "frustrate", "dislike", "hate"],
            "interests": ["interest", "hobby", "enjoy", "activity", "like", "love"],
            "interventions": ["intervention", "help", "assist", "support", "care"],
            "cultural": ["cultural", "background", "religion", "belief", "tradition"],
            "schedule": ["schedule", "routine", "daily", "time", "activity"],
            "breakfast": ["breakfast", "morning meal"],
            "lunch": ["lunch", "noon meal", "midday"],
            "dinner": ["dinner", "evening meal", "supper"],
            "activities": ["activity", "activities", "hobby", "do", "spend time"],
        }
        
        # ############################################################
        # # INFORMATION FILTERING - Extract only relevant data
        # ############################################################
        relevant_info = {}
        
        # Check for schedule-related queries
        if any(keyword in query_lower for keyword in ["schedule", "routine", "day", "daily"]):
            schedule_fields = ["Wake-Up", "Breakfast", "Morning Activities", "Lunch", 
                              "After Lunch Activities", "Afternoon Nap", "Dinner (5-6 PM)", 
                              "Turndown Time (6-7 PM)", "Sleep (7 PM)"]
            
            for field in schedule_fields:
                if field in persona_data:
                    relevant_info[field] = persona_data[field]
            return relevant_info
        
        # Check other specific categories
        for category, keywords in keyword_mapping.items():
            if any(keyword in query_lower for keyword in keywords):
                for key, value in persona_data.items():
                    key_lower = key.lower()
                    if any(keyword in key_lower for keyword in keywords):
                        relevant_info[key] = value
        
        # If no specific information was requested, return basic profile
        if not relevant_info:
            basic_fields = ["Name", "Age", "Level of Dementia", "Mobility", 
                           "Personality", "Conversational Ability"]
            for field in basic_fields:
                if field in persona_data:
                    relevant_info[field] = persona_data[field]
        
        return relevant_info
    
    def generate_response(self, query, persona_data, relevant_info):
        """Generate a natural language response using OpenAI."""
        # ############################################################
        # # PROMPT ENGINEERING - Create effective prompt for OpenAI
        # ############################################################
        try:
            # Create prompt for OpenAI
            persona_name = persona_data.get("Name", "the resident")
            prompt = f"""
            You are an assistant providing information about a person named {persona_name} at a care facility.
            
            Below is factual information about {persona_name}:
            {', '.join([f'{key}: {value}' for key, value in relevant_info.items()])}
            
            Based ONLY on the facts above, answer this question: "{query}"
            
            Keep your response concise, friendly, and only use information provided in the facts. 
            If information is not in the provided facts, say you don't have that information.
            Make the response conversational and helpful for a caregiver.
            """
            
            # ############################################################
            # # API CALL - Send request to OpenAI and get response
            # ############################################################
            # Call OpenAI API using the updated client format
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant providing factual information about care facility residents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"API Error: {str(e)}")
            return f"Error generating response. Please check your API key or try again later."
    
    def process_query(self, query):
        """Process the user query and generate a response."""
        # ############################################################
        # # QUERY PROCESSING - Main pipeline to handle user queries
        # ############################################################
        
        # Step 1: Identify which persona the query is about
        persona_name = self.identify_persona(query)
        
        if not persona_name:
            return "I'm not sure which resident you're asking about. Please specify a name in your question."
        
        # Step 2: Get all information about the persona
        persona_data = self.get_persona_info(persona_name)
        
        if not persona_data:
            return f"I don't have any information about {persona_name}."
        
        # Step 3: Extract relevant information based on the query
        relevant_info = self.extract_relevant_info(query, persona_data)
        
        # Step 4: Generate response
        response = self.generate_response(query, persona_data, relevant_info)
        
        return response

def main():
    # ############################################################
    # # INITIALIZATION - Start the chatbot application
    # ############################################################
    print("Initializing chatbot with OpenAI API...")
    print("API Key configured.")
    
    # File paths
    personas_file_path = r"PERSONA FILE PATH HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    activity_map_file_path = r"ACTIVITY MAP FILE PATH HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    
    try:
        # ############################################################
        # # CHATBOT CREATION - Initialize the chatbot with data
        # ############################################################
        # Initialize chatbot
        chatbot = PersonaChatbot(personas_file_path, activity_map_file_path)
        
        print("=" * 50)
        print("Persona Care Facility Chatbot")
        print("Type 'exit' to quit")
        print("=" * 50)
        
        # ############################################################
        # # CONVERSATION LOOP - Handle user input and generate responses
        # ############################################################
        while True:
            user_input = input("\nYour question: ")
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Thank you for using the chatbot. Goodbye!")
                break
            
            # ############################################################
            # # RESPONSE GENERATION - Process query and time the response
            # ############################################################
            start_time = time.time()
            response = chatbot.process_query(user_input)
            end_time = time.time()
            
            # ############################################################
            # # OUTPUT FORMATTING - Display response in readable format
            # ############################################################
            print("\nResponse:")
            # Pretty print the response with wrapping
            for line in textwrap.wrap(response, width=70):
                print(line)
            
            print(f"\n(Response generated in {end_time - start_time:.2f} seconds)")
    
    except Exception as e:
        # ############################################################
        # # ERROR HANDLING - Handle initialization errors
        # ############################################################
        print(f"Error initializing chatbot: {str(e)}")
        print("Please check if the CSV files exist at the specified paths.")

if __name__ == "__main__":
    main()
