// Code.gs

// Use the endpoint confirmed from your ngrok forward
const LOCAL_API_ENDPOINT =
  "https://53a1-2a02-aa7-464a-ae80-fde7-76b9-ef8f-1594.ngrok-free.app/predict"; // Make sure this is your CURRENT ngrok URL

/**
 * Builds the card shown when the add-on is opened without email context (Homepage).
 * @param {Object} e Event object (may be empty for homepage trigger).
 * @return {Card[]} Array containing the homepage card.
 */

function buildHomepageCard(e) {
  console.log("Building homepage card.");

  // Top section with main instruction
  const mainSection = CardService.newCardSection().addWidget(
    CardService.newTextParagraph().setText(
      "Please open an email to analyze it."
    )
  );

  // Spacer section (optional, makes things breathe a bit)
  const spacerSection1 = CardService.newCardSection().addWidget(
    CardService.newTextParagraph().setText("<br><br><br>")
  ); // Add vertical space

  // Separate section for centered image
  const imageSection = CardService.newCardSection().addWidget(
    CardService.newImage()
      .setImageUrl("https://cdn-icons-png.flaticon.com/256/873/873360.png")
      .setAltText("Information icon")
  );

  // Spacer section (optional, makes things breathe a bit)
  const spacerSection2 = CardService.newCardSection().addWidget(
    CardService.newTextParagraph().setText("<br><br><br>")
  ); // Add vertical space

  // Footer section
  const footerSection = CardService.newCardSection().addWidget(
    CardService.newTextParagraph().setText(
      "<div align='bottom'><font color='#888888' size='small'>If an email is already opened <br> Press the View button</font></div>"
    )
  );

  const card = CardService.newCardBuilder()
    .setHeader(CardService.newCardHeader().setTitle("No Email Detected"))
    .addSection(mainSection)
    .addSection(spacerSection1)
    .addSection(imageSection)
    .addSection(spacerSection2)
    .addSection(footerSection)
    .build();
  return [card];
}

/**
 * Callback function for contextual trigger when reading a message.
 * Builds the initial card with the "Analyze Email" button.
 * @param {Object} e The event object passed by the trigger.
 * @return {Card[]} Array containing the initial card.
 */

function buildPhishingCheckCard(e) {
  try {
    // Get the message ID AND access token from the event object
    const messageId = e.gmail.messageId;
    const accessToken = e.gmail.accessToken; // <-- Get the token here

    // Check if essential event data is present
    if (!messageId || !accessToken) {
      console.error("Missing messageId or accessToken in event object:", e);
      throw new Error("Could not get necessary context from Gmail.");
    }

    console.log("Building initial card for messageId: " + messageId);
    // Create the action that will be executed when the button is clicked.
    const analyzeAction = CardService.newAction()
      .setFunctionName("handleAnalyzeButtonClick")
      .setParameters({
        messageId: messageId,
        accessToken: accessToken, // <-- Pass the accessToken as a parameter
      });

    // Build the initial card (same as before)
    const cardSection = CardService.newCardSection()
      .addWidget(
        CardService.newTextParagraph().setText(
          "Click the button below to analyze this email for potential phishing."
        )
      )
      .addWidget(
        CardService.newButtonSet().addButton(
          CardService.newTextButton()
            .setText("Analyze Email")
            .setOnClickAction(analyzeAction)
        )
      );

    const imageSection = CardService.newCardSection().addWidget(
      CardService.newImage()
        .setImageUrl("https://cdn-icons-png.flaticon.com/256/873/873373.png")
        .setAltText("Information icon")
    );

    const card = CardService.newCardBuilder()
      .setHeader(CardService.newCardHeader().setTitle("Phishing Detector"))
      .addSection(cardSection)
      .addSection(imageSection)
      .build();
    return [card];
  } catch (error) {
    console.error("Error building initial card: " + error);
    return [buildErrorCard(error)];
  }
}

/**
 * Action handler function called when the "Analyze Email" button is clicked.
 * Fetches email details, calls the backend API, and returns a new card with the results.
 * @param {Object} actionEvent The event object passed from the button click action.
 * @return {ActionResponse} Response object to update the card UI.
 */
function handleAnalyzeButtonClick(actionEvent) {
  try {
    // --- Retrieve parameters passed from the button ---
    const messageId = actionEvent.parameters.messageId;
    const accessToken = actionEvent.parameters.accessToken; // <-- Retrieve the token

    // --- Validate parameters ---
    if (!messageId) {
      throw new Error("Message ID not found in action parameters.");
    }

    if (!accessToken) {
      // <-- Check if token was received
      throw new Error("Access Token not found in action parameters.");
    }

    console.log("handleAnalyzeButtonClick called for messageId: " + messageId);

    // --- 1. Set Access Token and Get Email Details ---
    GmailApp.setCurrentMessageAccessToken(accessToken); // <-- Use the passed token HERE
    const message = GmailApp.getMessageById(messageId); // <-- Now this should work
    const subject = message.getSubject();
    const body = message.getPlainBody();

    // --- 2. Prepare API Payload ---
    const rawText = (subject + " " + body).replace(/[\r\n]+/g, " ").trim();
    const payload = { raw_text: rawText };
    console.log("Payload: " + JSON.stringify(payload));

    // --- 3. Set API Call Options ---
    const options = {
      method: "post",
      contentType: "application/json",
      payload: JSON.stringify(payload),
      muteHttpExceptions: true,
    };

    // --- 4. Call Backend API ---
    console.log("Calling API: " + LOCAL_API_ENDPOINT);
    const response = UrlFetchApp.fetch(LOCAL_API_ENDPOINT, options);
    const responseCode = response.getResponseCode();
    const responseBody = response.getContentText();
    console.log("API Response Code: " + responseCode);

    // --- 5. Process API Response ---
    let resultText = "Analysis Result:";
    let isPhishing = false;
    if (responseCode === 200) {
      try {
        const predictionResult = JSON.parse(responseBody);
        if (predictionResult.hasOwnProperty("is_phishing")) {
          isPhishing = predictionResult.is_phishing;
          resultText = isPhishing
            ? "⚠️ Potential Phishing Detected!"
            : "✅ Looks Safe";
        } else {
          console.error(
            "API response missing 'is_phishing' key. Body: " + responseBody
          );
          resultText = "Error: Invalid API response format.";
        }
      } catch (err) {
        console.error(
          "Failed to parse API JSON response: " +
            err +
            ". Body: " +
            responseBody
        );
        resultText = "Error: Could not parse prediction result.";
      }
    } else {
      console.error(
        "API request failed. Code: " + responseCode + ", Body: " + responseBody
      );
      resultText =
        "Error: Could not contact detection service. (Code: " +
        responseCode +
        ")";

      if (responseCode === 404) {
        resultText += " Endpoint not found.";
      } else if (responseCode >= 500) {
        resultText += " Server error.";
      }
    }

    // --- 6. Determine Icon URL ---
    let iconURL = isPhishing
      ? "https://gds.baguette.engineering/icons/clear.png" // Warning icon
      : "https://gds.baguette.engineering/icons/check.png"; // Check icon

    let imageURL = isPhishing
      ? "https://cdn-icons-png.flaticon.com/256/873/873376.png"
      : "https://cdn-icons-png.flaticon.com/256/873/873375.png";

    // --- 7. Build RESULT Card Header ---
    const cardHeader = CardService.newCardHeader()
      .setTitle("Phishing Scan Result")
      .setImageUrl(iconURL);

    // --- 8. Build RESULT Card Sections and Widgets ---
    const cardSection = CardService.newCardSection()
      .addWidget(CardService.newTextParagraph().setText(resultText))
      .addWidget(
        CardService.newTextParagraph().setText(`<b>Subject:</b> ${subject}`)
      );

    const imageSection = CardService.newCardSection().addWidget(
      CardService.newImage()
        .setImageUrl(imageURL)
        .setAltText("Information icon")
    );

    // --- 9. Build the RESULT Card ---
    const resultCard = CardService.newCardBuilder()
      .setHeader(cardHeader)
      .addSection(cardSection)
      .addSection(imageSection)
      .build();

    // --- 10. Return ActionResponse to UPDATE the UI ---
    return CardService.newActionResponseBuilder()
      .setNavigation(CardService.newNavigation().updateCard(resultCard))
      .build();
  } catch (error) {
    console.error("Error within handleAnalyzeButtonClick function: " + error);
    console.error("Stack: " + error.stack);
    return CardService.newActionResponseBuilder()
      .setNavigation(
        CardService.newNavigation().updateCard(buildErrorCard(error))
      )
      .build();
  }
}

/**
 * Helper function to build a generic error card.
 * @param {Error} error The error object.
 * @return {Card} The Card object displaying the error.
 */

function buildErrorCard(error) {
  const errorCard = CardService.newCardBuilder()
    .setHeader(CardService.newCardHeader().setTitle("Add-on Error"))
    .addSection(
      CardService.newCardSection()
        .addWidget(
          CardService.newTextParagraph().setText(
            "An unexpected error occurred. Check logs for details."
          )
        )
        .addWidget(
          CardService.newTextParagraph().setText(
            error ? error.toString() : "Unknown error"
          )
        )
    )
    .build();
  return errorCard;
}
