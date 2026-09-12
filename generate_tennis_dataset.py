import pandas as pd

# Load the tennis match files
matches_2021 = pd.read_csv("data/atp_matches_2021.csv")
matches_2022 = pd.read_csv("data/atp_matches_2022.csv")
matches_2023 = pd.read_csv("data/atp_matches_2023.csv")
matches_2024 = pd.read_csv("data/atp_matches_2024.csv")

# Combine all matches
matches = pd.concat(
    [matches_2021, matches_2022, matches_2023, matches_2024],
    ignore_index=True
)

# Convert the tournament date to a real date
matches["tourney_date"] = pd.to_datetime(
    matches["tourney_date"],
    format="%Y%m%d"
)

# Sort matches by date
matches = matches.sort_values("tourney_date")

print("Total matches:", len(matches))
print(matches[[
    "tourney_date",
    "winner_name",
    "loser_name",
    "winner_rank",
    "loser_rank"
]].head())



def recent_win_rate(history):
    if len(history) == 0:
        return 0

    recent_matches = history[-10:]

    return sum(recent_matches) / len(recent_matches)

# Store previous results for each player
player_history = {}

tennis_data = []

for _, match in matches.iterrows():

    winner = match["winner_name"]
    loser = match["loser_name"]

    winner_rank = match["winner_rank"]
    loser_rank = match["loser_rank"]

    # Create history for new players
    if winner not in player_history:
        player_history[winner] = []

    if loser not in player_history:
        player_history[loser] = []

    # Calculate recent performance before the current match
    winner_recent_rate = recent_win_rate(player_history[winner])
    loser_recent_rate = recent_win_rate(player_history[loser])

        # Only create records from 2022 onwards
    # Both players must have enough previous matches
    if (
        match["tourney_date"].year >= 2022
        and len(player_history[winner]) >= 5
        and len(player_history[loser]) >= 5
        and pd.notna(winner_rank)
        and pd.notna(loser_rank)
    ):

        # Alternate Player A and Player B to balance the target classes
        if len(tennis_data) % 2 == 0:

            player_a = winner
            player_b = loser

            player_a_rank = winner_rank
            player_b_rank = loser_rank

            player_a_wins = 1

        else:

            player_a = loser
            player_b = winner

            player_a_rank = loser_rank
            player_b_rank = winner_rank

            player_a_wins = 0

        # Calculate ranking difference
        ranking_difference = player_b_rank - player_a_rank

        tennis_data.append({
            "player_a": player_a,
            "player_b": player_b,
            "player_a_ranking": player_a_rank,
            "player_b_ranking": player_b_rank,
            "ranking_difference": ranking_difference,
            "player_a_recent_win_rate": (
                winner_recent_rate if player_a == winner
                else loser_recent_rate
            ),
            "player_b_recent_win_rate": (
                loser_recent_rate if player_b == loser
                else winner_recent_rate
            ),
            "player_a_wins": player_a_wins
        })

    # Update history AFTER the match
    player_history[winner].append(1)
    player_history[loser].append(0)

# Convert the generated data into a DataFrame
tennis_df = pd.DataFrame(tennis_data)

print("Final records:", len(tennis_df))
print(tennis_df.head())

# Select a balanced sample for the classification project
class_1 = tennis_df[tennis_df["player_a_wins"] == 1].sample(
    n=600,
    random_state=42
)

class_0 = tennis_df[tennis_df["player_a_wins"] == 0].sample(
    n=600,
    random_state=42
)

# Combine both classes
tennis_df = pd.concat(
    [class_1, class_0],
    ignore_index=True
)

# Shuffle the final dataset
tennis_df = tennis_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nFinal classification dataset:")
print("Records:", len(tennis_df))
print(tennis_df["player_a_wins"].value_counts())

# Save the final dataset
output_file = "data/tennis_matches.csv"

tennis_df.to_csv(output_file, index=False)

print("\nDataset saved successfully.")
print("File:", output_file)

# Verify the saved file
saved_df = pd.read_csv(output_file)

print("\nSaved file verification:")
print("Saved records:", len(saved_df))
print(saved_df["player_a_wins"].value_counts())