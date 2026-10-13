import XCTest

final class QuantumCatIOSUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func testSetupSheetAndNewGameControlsExist() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-uiTestReset"]
        app.launch()

        XCTAssertTrue(app.buttons["setup-button"].waitForExistence(timeout: 6))
        app.buttons["Table setup"].tap()
        XCTAssertTrue(app.navigationBars["Table setup"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.buttons["New game"].exists)
        XCTAssertTrue(app.buttons["Done"].exists)
        app.buttons["Done"].tap()
        XCTAssertTrue(app.buttons["New game"].waitForExistence(timeout: 3))
    }

    func testCanTakeFirstLegalMove() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-uiTestReset"]
        app.launch()

        XCTAssertTrue(app.buttons["setup-button"].waitForExistence(timeout: 6))
        let firstMove = app.buttons.matching(NSPredicate(format: "label BEGINSWITH %@", "Discard ")).firstMatch
        XCTAssertTrue(firstMove.waitForExistence(timeout: 6))
        firstMove.tap()
    }

    func testDiscardAndBidUseHandRankTiles() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-uiTestReset"]
        app.launch()

        XCTAssertTrue(app.buttons["setup-button"].waitForExistence(timeout: 6))
        let discardMove = app.buttons.matching(NSPredicate(format: "label BEGINSWITH %@", "Discard ")).firstMatch
        XCTAssertTrue(discardMove.waitForExistence(timeout: 6))
        discardMove.tap()

        XCTAssertTrue(app.staticTexts["P0, choose your bid"].waitForExistence(timeout: 8))
        XCTAssertFalse(app.staticTexts["Legal moves"].exists)
        let bidMove = app.buttons.matching(NSPredicate(format: "label BEGINSWITH %@", "Bid ")).firstMatch
        XCTAssertTrue(bidMove.waitForExistence(timeout: 3))
        bidMove.tap()
    }

    func testBotRecommendationCanExpandAndApplyMove() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-uiTestReset"]
        app.launch()

        XCTAssertTrue(app.buttons["setup-button"].waitForExistence(timeout: 6))
        let recommendationToggle = app.buttons["bot-recommendation-toggle"]
        XCTAssertTrue(recommendationToggle.waitForExistence(timeout: 6))
        recommendationToggle.tap()

        XCTAssertTrue(app.staticTexts["Best move"].waitForExistence(timeout: 4))
        XCTAssertTrue(app.staticTexts["Second best"].exists)
        XCTAssertTrue(
            app.descendants(matching: .any)["bot-recommendation-score-0"].exists
        )
        XCTAssertTrue(
            app.descendants(matching: .any)["bot-recommendation-score-1"].exists
        )

        let useRecommendation = app.descendants(matching: .any)["bot-recommendation-use"]
        XCTAssertTrue(useRecommendation.waitForExistence(timeout: 4))
        useRecommendation.tap()

        XCTAssertTrue(app.staticTexts["P0, choose your bid"].waitForExistence(timeout: 8))
    }

    func testChampionCoachCanRevealScoresForEveryLegalMove() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-uiTestReset"]
        app.launch()

        let recommendationToggle = app.buttons["bot-recommendation-toggle"]
        XCTAssertTrue(recommendationToggle.waitForExistence(timeout: 6))
        recommendationToggle.tap()

        XCTAssertTrue(app.staticTexts["Champion preference · scores total 100"].exists)
        let remaining = app.descendants(matching: .any)["bot-recommendation-other-moves"]
        XCTAssertTrue(remaining.waitForExistence(timeout: 4))
        remaining.tap()
        XCTAssertTrue(
            app.descendants(matching: .any)["bot-recommendation-score-2"]
                .waitForExistence(timeout: 3)
        )
    }

    func testPlayedBoardCellIsMarkedAsCurrentTrick() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-uiTestReset"]
        app.launch()

        XCTAssertTrue(app.buttons["setup-button"].waitForExistence(timeout: 6))
        let discardMove = app.buttons.matching(NSPredicate(format: "label BEGINSWITH %@", "Discard ")).firstMatch
        XCTAssertTrue(discardMove.waitForExistence(timeout: 6))
        discardMove.tap()

        XCTAssertTrue(app.staticTexts["P0, choose your bid"].waitForExistence(timeout: 8))
        let bidMove = app.buttons.matching(NSPredicate(format: "label BEGINSWITH %@", "Bid ")).firstMatch
        XCTAssertTrue(bidMove.waitForExistence(timeout: 3))
        bidMove.tap()

        XCTAssertTrue(app.staticTexts["P0, choose a play"].waitForExistence(timeout: 8))
        let countedLegalMove = app.buttons.matching(NSPredicate(format: "label CONTAINS %@", "cards of rank")).firstMatch
        XCTAssertTrue(countedLegalMove.waitForExistence(timeout: 4))
        let playMove = app.buttons.matching(NSPredicate(format: "label MATCHES %@", "^[RBYG] [1-6]$")).firstMatch
        XCTAssertTrue(playMove.waitForExistence(timeout: 6))
        playMove.tap()

        let currentTrickCell = app.buttons.matching(NSPredicate(format: "label CONTAINS %@", "this trick")).firstMatch
        XCTAssertTrue(currentTrickCell.waitForExistence(timeout: 4))
    }

    func testBotAdvanceDoesNotShiftMainTurnPanel() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-uiTestReset"]
        app.launch()

        XCTAssertTrue(app.buttons["setup-button"].waitForExistence(timeout: 6))
        let panel = app.scrollViews["active-turn-panel"]
        XCTAssertTrue(panel.waitForExistence(timeout: 6))
        let initialTop = panel.frame.minY

        let firstMove = app.buttons.matching(NSPredicate(format: "label BEGINSWITH %@", "Discard ")).firstMatch
        XCTAssertTrue(firstMove.waitForExistence(timeout: 6))
        firstMove.tap()

        XCTAssertTrue(app.staticTexts["P0, choose your bid"].waitForExistence(timeout: 8))
        let finalTop = panel.frame.minY
        XCTAssertLessThanOrEqual(abs(finalTop - initialTop), 4, "Bot animation/return-to-human shifted the main table scroll from \(initialTop) to \(finalTop)")
    }

    func testMixedBotRosterAppearsInSetup() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-uiTestReset", "-uiTestMixedRoster"]
        app.launch()

        XCTAssertTrue(app.staticTexts["Quantum Cat"].waitForExistence(timeout: 6))
        app.buttons["Table setup"].tap()
        XCTAssertTrue(app.navigationBars["Table setup"].waitForExistence(timeout: 3))

        func revealSelectedBot(_ name: String) {
            let label = app.staticTexts["Selected: \(name)"]
            for _ in 0..<3 where !label.exists {
                app.swipeUp()
            }
            XCTAssertTrue(label.exists)
        }

        revealSelectedBot("Run1052 ML (previous champion)")
        revealSelectedBot("SetPool Distill")
        revealSelectedBot("Strict Q Head")
        app.buttons["Done"].tap()
    }

    func testBotOnlyBulkSimulationSummaryAppears() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-uiTestReset", "-uiTestBotOnlyBulk"]
        app.launch()

        XCTAssertTrue(app.staticTexts["Quantum Cat"].waitForExistence(timeout: 6))
        app.buttons["Table setup"].tap()
        XCTAssertTrue(app.navigationBars["Table setup"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.segmentedControls["bot-only-mode-picker"].exists)
        XCTAssertTrue(app.staticTexts["Games"].exists)
        app.buttons["setup-new-game-button"].tap()
        XCTAssertTrue(app.staticTexts["Bulk simulation"].waitForExistence(timeout: 8))
        XCTAssertTrue(app.staticTexts["5 games"].exists)
    }

    func testRun374BenchmarkSummaryAppears() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-uiTestReset", "-uiTestRun374Benchmark"]
        app.launch()

        XCTAssertTrue(app.staticTexts["Bulk simulation"].waitForExistence(timeout: 20))
        XCTAssertTrue(app.staticTexts["20 games"].exists)
        XCTAssertTrue(app.staticTexts["Any paradox"].exists)
        XCTAssertTrue(app.staticTexts["Seat paradox"].exists)
        XCTAssertTrue(app.staticTexts["Core ML"].exists)
        XCTAssertTrue(app.staticTexts["0.0%"].exists)
    }
}
