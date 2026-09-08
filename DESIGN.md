# Quantum Cat Web UI Design

## Product Framing

Quantum Cat is a thinking game first: the interface should make turn state, private options, public board claims, and bot strength visible without turning the screen into a tutorial. The first screen is the playable table with setup controls, not a landing page.

## Visual Direction

Use a focused tabletop operations feel: dark ink background, warm paper panels, saturated suit colors, and compact controls. The main visual object is the color-by-rank claim board, supported by seat strips and an action rail. The tone should feel deliberate and competitive rather than generic SaaS.

## Safe Choices And Bold Bets

Safe choices:
- Keep all legal actions visible as buttons.
- Keep the current player, phase, trick count, predictions, and scores persistent.
- Use restrained panels with 8px radii, clear focus states, and high contrast text.

Bold bets:
- Treat the board as the primary action surface during play: legal color/rank plays can be made by clicking the grid.
- Use suit-color row accents and compact seat badges so a player can scan state at game speed.

## Type System

Use system sans-serif for all UI text. Keep large type only for the active player and game title. Use tabular numerals for ranks, tricks, predictions, and scores so the board and scoreboard stay stable.

## Color System

Base colors:
- Ink: `#151716`
- Panel: `#f6f0e4`
- Panel muted: `#e9dfcd`
- Text: `#1d211f`
- Muted text: `#68706a`
- Border: `#d4c6ad`

Suit colors:
- Red: `#c63d32`
- Blue: `#286fa3`
- Yellow: `#d1a22a`
- Green: `#3f8f62`

Use the suit colors for board rows and card/action chips, but do not let a single hue dominate the whole UI.

## Layout And Spacing

Use a three-zone layout on desktop: left sidebar for setup/status/phase, center stage for the current human task, and right rail for public state. On mobile, stack setup, phase/status, current task, players, board, and log in that order. Spacing should use a 4px base rhythm with 8px, 12px, 16px, and 24px steps.

## Components And Interaction Rules

- The center stage must answer "what do I do now?" with a large task title and one plain hint.
- Buttons use concise labels, not explanatory copy.
- Legal moves are grouped by phase or suit so players do not have to parse one long undifferentiated list.
- Action buttons are stable-height and never resize the board.
- Board cells have fixed aspect ratio and clear disabled, legal, occupied, and blocked states.
- Seat rows show role, turn marker, tricks, prediction, paradox state, and final score.
- Current human private hand appears only in the center action stage.
- Bot turns auto-advance after human actions, with every bot decision recorded in the log.

## Mobile And Accessibility Notes

The board must remain horizontally usable on mobile with compact rank columns and no overlapping text. All interactive controls need visible focus outlines. Color is paired with labels or rank text, so suit state is not color-only.
