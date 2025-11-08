#  Usage --> python cli_app.py -o "Bangalore, India" -d "Krabi, Thailand" -s 2024-05-01 -e 2024-05-10 -i "2 adults who love swimming, dancing, hiking, shopping, food, water sports adventures, rock climbing"
# python cli_app.py -o "Singapore," -d "Hongkong, China" -s 2025-11-12 -e 2025-11-16 -i "2 adults who love swimming, dancing, shopping,,we are statying as hotel Bahunia, food only at Indian restaurents, victoria hill adventures"
from crewai import Crew, LLM
from trip_agents import TripAgents
from trip_tasks import TripTasks
from datetime import datetime, timedelta
import argparse
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = (
    "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"
)


class TripCrew:
    def __init__(self, origin, cities, date_range, interests):
        self.cities = cities
        self.origin = origin
        self.interests = interests
        self.date_range = date_range
        # self.llm = LLM(model="groq/deepseek-r1-distill-llama-70b")
        # self.llm = LLM(
        #     model="gpt-3.5-turbo",  # or "gpt-4"
        #     api_key=os.getenv("OPENAI_API_KEY"),  # just your key
        # )
        self.llm = LLM(
            model="openrouter/tngtech/deepseek-r1t2-chimera:free",  # ✅ add 'openrouter/' prefix
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        # self.llm = LLM(model="gemini/gemini-2.5-flash")

    def run(self):
        try:
            agents = TripAgents(llm=self.llm)
            tasks = TripTasks()

            city_selector_agent = agents.city_selection_agent()
            local_expert_agent = agents.local_expert()
            travel_concierge_agent = agents.travel_concierge()

            identify_task = tasks.identify_task(
                city_selector_agent,
                self.origin,
                self.cities,
                self.interests,
                self.date_range,
            )

            gather_task = tasks.gather_task(
                local_expert_agent, self.origin, self.interests, self.date_range
            )

            plan_task = tasks.plan_task(
                travel_concierge_agent, self.origin, self.interests, self.date_range
            )

            crew = Crew(
                agents=[
                    city_selector_agent,
                    local_expert_agent,
                    travel_concierge_agent,
                ],
                tasks=[identify_task, gather_task, plan_task],
                verbose=True,
            )

            result = crew.kickoff()
            return result
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None


def validate_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format. Please use YYYY-MM-DD")


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Check if required API keys are set
    required_keys = ["OPENAI_API_KEY", "SERPER_API_KEY", "BROWSERLESS_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        print(
            f"Error: Missing required environment variables: {', '.join(missing_keys)}"
        )
        print("Please set them in your .env file or environment.")
        return

    parser = argparse.ArgumentParser(description="AI Travel Planner")

    parser.add_argument(
        "--origin",
        "-o",
        type=str,
        required=True,
        help='Your current location (e.g., "San Mateo, CA")',
    )

    parser.add_argument(
        "--destination",
        "-d",
        type=str,
        required=True,
        help='Destination city and country (e.g., "Bali, Indonesia")',
    )

    parser.add_argument(
        "--start-date",
        "-s",
        type=validate_date,
        required=True,
        help="Start date of your trip (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end-date",
        "-e",
        type=validate_date,
        required=True,
        help="End date of your trip (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--interests",
        "-i",
        type=str,
        required=True,
        help='Your interests and trip details (e.g., "2 adults who love swimming, dancing, hiking")',
    )

    args = parser.parse_args()

    # Validate dates
    if args.end_date <= args.start_date:
        print("Error: End date must be after start date")
        return

    # Format date range as string
    date_range = f"{args.start_date} to {args.end_date}"

    print("\n🏖️ VacAIgent - AI Travel Planner")
    print("================================")
    print(f"\nPlanning your trip...")
    print(f"From: {args.origin}")
    print(f"To: {args.destination}")
    print(f"Dates: {date_range}")
    print(f"Interests: {args.interests}")
    print(
        "\nThis may take a few minutes. Please wait while our AI agents work on your perfect trip...\n"
    )

    trip_crew = TripCrew(args.origin, args.destination, date_range, args.interests)
    result = trip_crew.run()

    if result:
        print("\n✨ Your Trip Plan ✨")
        print("===================\n")
        print(result)
    else:
        print("\n❌ Failed to generate trip plan. Please try again.")


if __name__ == "__main__":
    main()
