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
        XCTAssertTrue(app.staticTexts["Selected: Champion ML"].exists)
        XCTAssertTrue(app.staticTexts["Selected: SetPool Distill"].exists)
        if !app.staticTexts["Selected: Strict Q Head"].exists {
            app.swipeUp()
        }
        XCTAssertTrue(app.staticTexts["Selected: Strict Q Head"].exists)
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
