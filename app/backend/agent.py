from llama_index.llms.gemini import Gemini
import os
import logging

class Agent:
    def __init__(self):
        # More specific RAG keywords - only for specific information queries
        self.rag_keywords = [
            "फिस", "शुल्क", "पैसा", "रुपैयाँ", "कति", "कोर्स", "भर्ना", "प्रवेश", "परीक्षा", "सिलेबस",
            "विषय", "समय", "अवधि", "वर्ष", "सेमेस्टर", "कलेज", "विश्वविद्यालय", "डिग्री", "सर्टिफिकेट",
            "fee", "cost", "price", "course", "admission", "exam", "syllabus", "duration", "college", "university"
        ]

    def classify(self, text: str) -> str:
        # Smart classification - use RAG only for information queries, normal conversation for greetings/casual chat
        lowered = text.lower()

        # Check for casual greetings and simple conversations
        casual_patterns = [
            "नमस्ते", "नमस्कार", "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            "कस्तो छ", "how are you", "के छ", "के गर्दै", "what's up",
            "को हो", "who are you", "तपाईं को हुनुहुन्छ", "with whom am i talking",
            "तपाईंको नाम के हो", "what is your name", "who am i talking to"
        ]

        # Check for thank you / goodbye patterns
        goodbye_patterns = [
            "धन्यवाद", "thank you", "thanks", "bye", "अलविदा", "goodbye", "see you",
            "बाई", "टाटा", "फेरि भेटौंला", "see you later", "good bye"
        ]

        # Check if it's a simple greeting/casual conversation (short and matches patterns)
        is_casual = False
        if len(lowered.split()) <= 5:  # Short phrases
            for pattern in casual_patterns:
                if pattern in lowered:
                    is_casual = True
                    break

        # Check if it's a goodbye/thank you message
        is_goodbye = False
        for pattern in goodbye_patterns:
            if pattern in lowered:
                is_goodbye = True
                break

        # Check for information-seeking indicators
        rag_indicators = 0
        for kw in self.rag_keywords:
            if kw in lowered:
                rag_indicators += 1

        # Question words that indicate information seeking (excluding personal questions)
        question_words = ["कति", "के", "कसरी", "कहाँ", "कुन", "कहिले", "what", "how", "where", "when", "which", "why", "कुन"]
        has_question_word = any(q in lowered for q in question_words)

        # Personal questions that should use normal conversation
        personal_questions = ["who are you", "को हो", "तपाईं को", "your name", "तपाईंको नाम"]
        is_personal_question = any(p in lowered for p in personal_questions)

        # Information seeking phrases
        info_phrases = ["बारे", "about", "जानकारी", "information", "बताउनुस्", "tell me", "चाहिन्छ", "need", "खोज्दै", "looking for"]
        has_info_phrase = any(phrase in lowered for phrase in info_phrases)

        # Decision logic
        if is_goodbye:
            return "goodbye"

        if is_casual and rag_indicators == 0 and not has_question_word:
            return "normal"

        if is_personal_question:
            return "normal"

        # Use RAG if there are education keywords OR question words OR info-seeking phrases
        if rag_indicators > 0 or has_question_word or has_info_phrase:
            return "rag"

        # Default to normal conversation for unclear cases
        return "normal"

    def goodbye_conversation(self, text: str) -> str:
        """Handle goodbye/thank you messages"""
        goodbye_responses = [
            "धन्यवाद! काठमाडौं विश्वविद्यालयको भर्ना सहायकसँग कुरा गर्नुभएकोमा खुसी लाग्यो। फेरि कुनै प्रश्न भए सम्पर्क गर्नुहोला!",
            "तपाईंलाई धन्यवाद! KUSOE बारे थप जानकारीको लागि जहिले पनि सम्पर्क गर्न सक्नुहुन्छ। शुभकामना!",
            "धन्यवाद! काठमाडौं विश्वविद्यालयमा तपाईंको भविष्य उज्ज्वल होस्। फेरि भेटौंला!"
        ]
        import random
        return random.choice(goodbye_responses)

    def normal_conversation(self, text: str) -> str:
        """Handle normal conversation using Gemini LLM"""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logging.error("GEMINI_API_KEY not found in environment")
                return "माफ गर्नुहोस्, म अहिले कुराकानी गर्न सक्दिन।"

            llm = Gemini(
                model_name="models/gemini-2.5-flash",
                temperature=0.5,  # Balanced creativity
                top_p=0.8,
                top_k=30,
                max_tokens=60  # Limit to 1 sentence
            )

            prompt = (
                "तपाईं काठमाडौं विश्वविद्यालयको भर्ना सहायक हुनुहुन्छ। "
                "प्रयोगकर्तासँग मित्रवत् कुराकानी गर्नुहोस् र उनीहरूलाई भर्ना सम्बन्धी प्रश्न सोध्न प्रोत्साहन दिनुहोस्। "
                "केवल एक वाक्यमा मात्र छोटो जवाफ दिनुहोस्। "
                "यदि प्रयोगकर्ताले सामान्य कुराकानी गरेको छ भने उनीहरूलाई भर्ना, कोर्स, फिस, वा अध्ययन सम्बन्धी प्रश्न सोध्न भन्नुहोस्। "
                "यदि प्रयोगकर्ताले काठमाडौं विश्वविद्यालय बाहिरका विषयहरू (जस्तै मौसम, राजनीति, खेल, फिल्म) बारे सोधेको छ भने 'माफ गर्नुहोस्, मसँग यस बारेमा जानकारी छैन। के तपाईं हाम्रो ग्राहक प्रतिनिधिसँग कुरा गर्न चाहनुहुन्छ?' भन्नुहोस्।\n\n"
                f"प्रयोगकर्ता: {text}\n"
                "तपाईं:"
            )

            response = llm.complete(prompt)
            return str(response).strip()

        except Exception as e:
            logging.error(f"Error in normal conversation: {e}")
            return "नमस्ते! म काठमाडौं विश्वविद्यालयको भर्ना सहायक हुँ। तपाईंले भर्ना, कोर्स, वा फिस सम्बन्धी प्रश्न सोध्न सक्नुहुन्छ।"