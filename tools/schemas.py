"""Auto-generated Pydantic input schemas for all BFCL tools."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal, Union

# ─── PostingAPI ──────────


# ─── GorillaFileSystem ──────────

class CatInput(BaseModel):
    file_name: str = Field(description="The name of the file from current directory to display. No path is allowed. ")

class CdInput(BaseModel):
    folder: str = Field(description="The folder of the directory to change to. You can only change one folder at a time. ")

class CpInput(BaseModel):
    destination: str = Field(description="The destination name to copy the file or directory to. If the destination is a directory, the source will be copied into this directory. No file paths allowed. ")
    source: str = Field(description="The name of the file or directory to copy.")

class DiffInput(BaseModel):
    file_name1: str = Field(description="The name of the first file in current directory.")
    file_name2: str = Field(description="The name of the second file in current directorry. ")

class DuInput(BaseModel):
    human_readable: Optional[bool] = Field(default=None, description="If True, returns the size in human-readable format (e.g., KB, MB). ")

class EchoInput(BaseModel):
    content: str = Field(description="The content to write or display.")
    file_name: Optional[str] = Field(default=None, description="The name of the file at current directory to write the content to. Defaults to None. ")

class FindInput(BaseModel):
    name: Optional[str] = Field(default=None, description="The name of the file or directory to search for. If None, all items are returned. ")
    path: Optional[str] = Field(default=None, description="The directory path to start the search. Defaults to the current directory (\".\").")

class GrepInput(BaseModel):
    file_name: str = Field(description="The name of the file to search. No path is allowed and you can only perform on file at local directory.")
    pattern: str = Field(description="The pattern to search for. ")

class LsInput(BaseModel):
    a: Optional[bool] = Field(default=None, description="Show hidden files and directories. Defaults to False. ")

class MkdirInput(BaseModel):
    dir_name: str = Field(description="The name of the new directory at current directory. You can only create directory at current directory.")

class MvInput(BaseModel):
    destination: str = Field(description="The destination name to move the file or directory to. Destination must be local to the current directory and cannot be a path. If destination is not an existing directory like when renaming something, destination is the new file name. ")
    source: str = Field(description="Source name of the file or directory to move. Source must be local to the current directory.")

class RmInput(BaseModel):
    file_name: str = Field(description="The name of the file or directory to remove. ")

class RmdirInput(BaseModel):
    dir_name: str = Field(description="The name of the directory to remove. Directory must be local to the current directory. ")

class SortInput(BaseModel):
    file_name: str = Field(description="The name of the file appeared at current directory to sort. ")

class TailInput(BaseModel):
    file_name: str = Field(description="The name of the file to display. No path is allowed and you can only perform on file at local directory.")
    lines: Optional[int] = Field(default=None, description="The number of lines to display from the end of the file. Defaults to 10. ")

class TouchInput(BaseModel):
    file_name: str = Field(description="The name of the new file in the current directory. file_name is local to the current directory and does not allow path.")

class WcInput(BaseModel):
    file_name: str = Field(description="Name of the file of current directory to perform wc operation on.")
    mode: Optional[str] = Field(default=None, description="Mode of operation ('l' for lines, 'w' for words, 'c' for characters). ")

# ─── MathAPI ──────────

class AbsoluteValueInput(BaseModel):
    number: str = Field(description="The number to calculate the absolute value of. ")

class AddInput(BaseModel):
    a: str = Field(description="First number.")
    b: str = Field(description="Second number. ")

class DivideInput(BaseModel):
    a: str = Field(description="Numerator.")
    b: str = Field(description="Denominator. ")

class ImperialSiConversionInput(BaseModel):
    unit_in: str = Field(description="Unit of the input value.")
    unit_out: str = Field(description="Unit to convert the value to. ")
    value: str = Field(description="Value to be converted.")

class LogarithmInput(BaseModel):
    base: str = Field(description="The base of the logarithm.")
    precision: int = Field(description="Desired precision for the result. ")
    value: str = Field(description="The number to compute the logarithm of.")

class MaxValueInput(BaseModel):
    numbers: List = Field(description="List of numbers to find the maximum from. ")

class MeanInput(BaseModel):
    numbers: List = Field(description="List of numbers to calculate the mean of. ")

class MinValueInput(BaseModel):
    numbers: List = Field(description="List of numbers to find the minimum from. ")

class MultiplyInput(BaseModel):
    a: str = Field(description="First number.")
    b: str = Field(description="Second number. ")

class PercentageInput(BaseModel):
    part: str = Field(description="The part value.")
    whole: str = Field(description="The whole value. ")

class PowerInput(BaseModel):
    base: str = Field(description="The base number.")
    exponent: str = Field(description="The exponent. ")

class RoundNumberInput(BaseModel):
    decimal_places: Optional[int] = Field(default=None, description="The number of decimal places to round to. Defaults to 0. ")
    number: str = Field(description="The number to round.")

class SiUnitConversionInput(BaseModel):
    unit_in: str = Field(description="Unit of the input value.")
    unit_out: str = Field(description="Unit to convert the value to. ")
    value: str = Field(description="Value to be converted.")

class SquareRootInput(BaseModel):
    number: str = Field(description="The number to calculate the square root of.")
    precision: int = Field(description="Desired precision for the result. ")

class StandardDeviationInput(BaseModel):
    numbers: List = Field(description="List of numbers to calculate the standard deviation of. ")

class SubtractInput(BaseModel):
    a: str = Field(description="Number to subtract from.")
    b: str = Field(description="Number to subtract. ")

class SumValuesInput(BaseModel):
    numbers: List = Field(description="List of numbers to sum. ")

# ─── MessageAPI ──────────

class AddContactInput(BaseModel):
    user_name: str = Field(description="User name of contact to be added.")

class DeleteMessageInput(BaseModel):
    message_id: Optional[int] = Field(default=None, description="ID of the message to be deleted.")
    receiver_id: str = Field(description="User ID of the user to send the message to.")

class GetUserIdInput(BaseModel):
    user: str = Field(description="User name of the user. ")

class MessageLoginInput(BaseModel):
    user_id: str = Field(description="User ID of the user to log in. ")

class SearchMessagesInput(BaseModel):
    keyword: str = Field(description="The keyword to search for in messages.")

class SendMessageInput(BaseModel):
    message: str = Field(description="Message to be sent.")
    receiver_id: str = Field(description="User ID of the user to send the message to.")




class AuthenticateTwitterInput(BaseModel):
    """Input schema for authenticate_twitter method."""
    username: str = Field(..., description="Username of the user.")
    password: str = Field(..., description="Password of the user.")


class CommentInput(BaseModel):
    """Input schema for comment method."""
    tweet_id: int = Field(..., description="ID of the tweet to comment on.")
    comment_content: str = Field(..., description="Content of the comment.")


class FollowUserInput(BaseModel):
    """Input schema for follow_user method."""
    username_to_follow: str = Field(..., description="Username of the user to follow.")


class GetTweetInput(BaseModel):
    """Input schema for get_tweet method."""
    tweet_id: int = Field(..., description="ID of the tweet to retrieve.")


class GetTweetCommentsInput(BaseModel):
    """Input schema for get_tweet_comments method."""
    tweet_id: int = Field(..., description="ID of the tweet to retrieve comments for.")


class GetUserStatsInput(BaseModel):
    """Input schema for get_user_stats method."""
    username: str = Field(..., description="Username of the user to get statistics for.")


class GetUserTweetsInput(BaseModel):
    """Input schema for get_user_tweets method."""
    username: str = Field(..., description="Username of the user whose tweets to retrieve.")


class MentionInput(BaseModel):
    """Input schema for mention method."""
    tweet_id: int = Field(..., description="ID of the tweet where users are mentioned.")
    mentioned_usernames: List[str] = Field(..., description="List of usernames to be mentioned.")


class PostTweetInput(BaseModel):
    """Input schema for post_tweet method."""
    content: str = Field(..., description="Content of the tweet.")
    tags: List[str] = Field(default=[], description="List of tags for the tweet. Tag name should start with #. This is only relevant if the user wants to add tags to the tweet.")
    mentions: List[str] = Field(default=[], description="List of users mentioned in the tweet. Mention name should start with @. This is only relevant if the user wants to add mentions to the tweet.")


class RetweetInput(BaseModel):
    """Input schema for retweet method."""
    tweet_id: int = Field(..., description="ID of the tweet to retweet.")


class SearchTweetsInput(BaseModel):
    """Input schema for search_tweets method."""
    keyword: str = Field(..., description="Keyword to search for in the content of the tweets.")


class UnfollowUserInput(BaseModel):
    """Input schema for unfollow_user method."""
    username_to_unfollow: str = Field(..., description="Username of the user to unfollow.")

# ─── TicketAPI ──────────



class CloseTicketInput(BaseModel):
    """Input schema for close_ticket method."""
    ticket_id: int = Field(..., description="ID of the ticket to be closed.")


class CreateTicketInput(BaseModel):
    """Input schema for create_ticket method."""
    title: str = Field(..., description="Title of the ticket.")
    description: str = Field('', description="Description of the ticket. Defaults to an empty string.")
    priority: int = Field(1, description="Priority of the ticket, from 1 to 5. Defaults to 1. 5 is the highest priority.")


class EditTicketInput(BaseModel):
    """Input schema for edit_ticket method."""
    ticket_id: int = Field(..., description="ID of the ticket to be changed.")
    updates: Dict[str, Any] = Field(..., description="Dictionary containing the fields to be updated. - title (str) : [Optional] New title for the ticket.")


class GetTicketInput(BaseModel):
    """Input schema for get_ticket method."""
    ticket_id: int = Field(..., description="ID of the ticket to retrieve.")


class GetUserTicketsInput(BaseModel):
    """Input schema for get_user_tickets method."""
    status: Optional[str] = Field(None, description="Status to filter tickets by. If None, return all tickets.")


class ResolveTicketInput(BaseModel):
    """Input schema for resolve_ticket method."""
    ticket_id: int = Field(..., description="ID of the ticket to be resolved.")
    resolution: str = Field(..., description="Resolution details for the ticket.")


class TicketLoginInput(BaseModel):
    """Input schema for ticket_login method."""
    username: str = Field(..., description="Username of the user.")
    password: str = Field(..., description="Password of the user.")

# ─── TradingBot ──────────


class AddToWatchlistInput(BaseModel):
    """Input schema for add_to_watchlist."""
    stock: str = Field(..., description="the stock symbol to add to the watchlist.")

class CancelOrderInput(BaseModel):
    """Input schema for cancel_order."""
    order_id: int = Field(..., description="ID of the order to cancel.")

class FilterStocksByPriceInput(BaseModel):
    """Input schema for filter_stocks_by_price."""
    stocks: List[str] = Field(..., description="List of stock symbols to filter.")
    min_price: float = Field(..., description="Minimum stock price.")
    max_price: float = Field(..., description="Maximum stock price.")

class FundAccountInput(BaseModel):
    """Input schema for fund_account."""
    amount: float = Field(..., description="Amount to fund the account with.")

class GetAvailableStocksInput(BaseModel):
    """Input schema for get_available_stocks."""
    sector: str = Field(..., description="The sector to retrieve stocks from (e.g., 'Technology').")

class GetOrderDetailsInput(BaseModel):
    """Input schema for get_order_details."""
    order_id: int = Field(..., description="ID of the order.")

class GetStockInfoInput(BaseModel):
    """Input schema for get_stock_info."""
    symbol: str = Field(..., description="Symbol that uniquely identifies the stock.")

class GetSymbolByNameInput(BaseModel):
    """Input schema for get_symbol_by_name."""
    name: str = Field(..., description="Name of the company.")

class GetTransactionHistoryInput(BaseModel):
    """Input schema for get_transaction_history."""
    start_date: Optional[str] = Field(None, description="Start date for the history (format: 'YYYY-MM-DD').")
    end_date: Optional[str] = Field(None, description="End date for the history (format: 'YYYY-MM-DD').")

class MakeTransactionInput(BaseModel):
    """Input schema for make_transaction."""
    account_id: int = Field(..., description="ID of the account.")
    xact_type: str = Field(..., description="Transaction type (deposit or withdrawal).")
    amount: float = Field(..., description="Amount to deposit or withdraw.")

class NotifyPriceChangeInput(BaseModel):
    """Input schema for notify_price_change."""
    stocks: List[str] = Field(..., description="List of stock symbols to check.")
    threshold: float = Field(..., description="Percentage change threshold to trigger a notification.")

class PlaceOrderInput(BaseModel):
    """Input schema for place_order."""
    order_type: str = Field(..., description="Type of the order (Buy/Sell).")
    symbol: str = Field(..., description="Symbol of the stock to trade.")
    price: float = Field(..., description="Price at which to place the order.")
    amount: int = Field(..., description="Number of shares to trade.")

class RemoveStockFromWatchlistInput(BaseModel):
    """Input schema for remove_stock_from_watchlist."""
    symbol: str = Field(..., description="Symbol of the stock to remove.")

class TradingLoginInput(BaseModel):
    """Input schema for trading_login."""
    username: str = Field(..., description="Username for authentication.")
    password: str = Field(..., description="Password for authentication.")

class UpdateMarketStatusInput(BaseModel):
    """Input schema for update_market_status."""
    current_time_str: str = Field(..., description="Current time in HH:MM AM/PM format.")

class UpdateStockPriceInput(BaseModel):
    """Input schema for update_stock_price."""
    symbol: str = Field(..., description="Symbol of the stock to update.")
    new_price: float = Field(..., description="New price of the stock.")

# ─── TravelBooking ──────────



class AuthenticateTravelInput(BaseModel):
    """Input schema for authenticate_travel method."""
    client_id: str = Field(..., description="The client applications client_id supplied by App Management")
    client_secret: str = Field(..., description="The client applications client_secret supplied by App Management")
    refresh_token: str = Field(..., description="The refresh token obtained from the initial authentication")
    grant_type: Literal["read_write", "read", "write"] = Field(..., description="The grant type of the authentication request. Here are the options: read_write, read, write")
    user_first_name: str = Field(..., description="The first name of the user")
    user_last_name: str = Field(..., description="The last name of the user")


class BookFlightInput(BaseModel):
    """Input schema for book_flight method."""
    access_token: str = Field(..., description="The access token obtained from the authenticate")
    card_id: str = Field(..., description="The ID of the credit card to use for the booking")
    travel_date: str = Field(..., description="The date of the travel in the format YYYY-MM-DD")
    travel_from: str = Field(..., description="The location the travel is from")
    travel_to: str = Field(..., description="The location the travel is to")
    travel_class: str = Field(..., description="The class of the travel")
    travel_cost: float = Field(..., description="The cost of the travel")


class CancelBookingInput(BaseModel):
    """Input schema for cancel_booking method."""
    access_token: str = Field(..., description="The access token obtained from the authenticate")
    booking_id: str = Field(..., description="The ID of the booking")


class ComputeExchangeRateInput(BaseModel):
    """Input schema for compute_exchange_rate method."""
    base_currency: Literal["USD", "RMB", "EUR", "JPY", "GBP", "CAD", "AUD", "INR", "RUB", "BRL", "MXN"] = Field(..., description="The base currency.")
    target_currency: Literal["USD", "RMB", "EUR", "JPY", "GBP", "CAD", "AUD", "INR", "RUB", "BRL", "MXN"] = Field(..., description="The target currency.")
    value: float = Field(..., description="The value to convert")


class ContactCustomerSupportInput(BaseModel):
    """Input schema for contact_customer_support method."""
    booking_id: str = Field(..., description="The ID of the booking")
    message: str = Field(..., description="The message to send to customer support")


class GetBudgetFiscalYearInput(BaseModel):
    """Input schema for get_budget_fiscal_year method."""
    lastModifiedAfter: Optional[str] = Field(None, description="Use this field if you only want Fiscal Years that were changed after the supplied date. The supplied date will be interpreted in the UTC time zone. If lastModifiedAfter is not supplied, the service will return all Fiscal Years, regardless of modified date. Example: 2016-03-29T16:12:20. Return in the format of YYYY-MM-DDTHH:MM:SS.")
    includeRemoved: Optional[str] = Field(None, description="If true, the service will return all Fiscal Years, including those that were previously removed. If not supplied, this field defaults to false.")


class GetCreditCardBalanceInput(BaseModel):
    """Input schema for get_credit_card_balance method."""
    access_token: str = Field(..., description="The access token obtained from the authenticate")
    card_id: str = Field(..., description="The ID of the credit card")


class GetFlightCostInput(BaseModel):
    """Input schema for get_flight_cost method."""
    travel_from: str = Field(..., description="The 3 letter code of the departing airport")
    travel_to: str = Field(..., description="The 3 letter code of the arriving airport")
    travel_date: str = Field(..., description="The date of the travel in the format 'YYYY-MM-DD'")
    travel_class: Literal["economy", "business", "first"] = Field(..., description="The class of the travel. Options are: economy, business, first.")


class GetNearestAirportByCityInput(BaseModel):
    """Input schema for get_nearest_airport_by_city method."""
    location: Literal["Rivermist", "Stonebrook", "Maplecrest", "Silverpine", "Shadowridge", "London", "Paris", "Sunset Valley", "Oakendale", "Willowbend", "Crescent Hollow", "Autumnville", "Pinehaven", "Greenfield", "San Francisco", "Los Angeles", "New York", "Chicago", "Boston", "Beijing", "Hong Kong", "Rome", "Tokyo"] = Field(..., description="The name of the location.")


class PurchaseInsuranceInput(BaseModel):
    """Input schema for purchase_insurance method."""
    access_token: str = Field(..., description="The access token obtained from the authenticate")
    insurance_type: str = Field(..., description="The type of insurance to purchase")
    insurance_cost: float = Field(..., description="The cost of the insurance")
    booking_id: str = Field(..., description="The ID of the booking")
    card_id: str = Field(..., description="The ID of the credit card to use for the")


class RegisterCreditCardInput(BaseModel):
    """Input schema for register_credit_card method."""
    access_token: str = Field(..., description="The access token obtained from the authenticate method")
    card_number: str = Field(..., description="The credit card number")
    expiration_date: str = Field(..., description="The expiration date of the credit card in the format MM/YYYY")
    cardholder_name: str = Field(..., description="The name of the cardholder")
    card_verification_number: int = Field(..., description="The card verification number")


class RetrieveInvoiceInput(BaseModel):
    """Input schema for retrieve_invoice method."""
    access_token: str = Field(..., description="The access token obtained from the authenticate")
    booking_id: Optional[str] = Field(None, description="The ID of the booking")
    insurance_id: Optional[str] = Field(None, description="The ID of the insurance")


class SetBudgetLimitInput(BaseModel):
    """Input schema for set_budget_limit method."""
    access_token: str = Field(..., description="The access token obtained from the authentication process or initial configuration.")
    budget_limit: float = Field(..., description="The budget limit to set in USD")


class VerifyTravelerInformationInput(BaseModel):
    """Input schema for verify_traveler_information method."""
    first_name: str = Field(..., description="The first name of the traveler")
    last_name: str = Field(..., description="The last name of the traveler")
    date_of_birth: str = Field(..., description="The date of birth of the traveler in the format YYYY-MM-DD")
    passport_number: str = Field(..., description="The passport number of the traveler")

# ─── VehicleControl ──────────



class ActivateParkingBrakeInput(BaseModel):
    """Input schema for activateParkingBrake."""
    mode: Literal["engage", "release"] = Field(..., description="The mode to set.")


class AdjustClimateControlInput(BaseModel):
    """Input schema for adjustClimateControl."""
    temperature: float = Field(..., description="The temperature to set in degree. Default to be celsius.")
    unit: Literal["celsius", "fahrenheit"] = Field(default="celsius", description="The unit of temperature.")
    fanSpeed: int = Field(default=50, description="The fan speed to set from 0 to 100. Default is 50.")
    mode: Literal["auto", "cool", "heat", "defrost"] = Field(default="auto", description="The climate mode to set.")


class DisplayCarStatusInput(BaseModel):
    """Input schema for displayCarStatus."""
    option: Literal["fuel", "battery", "doors", "climate", "headlights", "parkingBrake", "brakePadle", "engine"] = Field(..., description="The option to display.")


class DisplayLogInput(BaseModel):
    """Input schema for display_log."""
    messages: List[Any] = Field(..., description="The list of messages to display.")


class EstimateDistanceInput(BaseModel):
    """Input schema for estimate_distance."""
    cityA: str = Field(..., description="The zipcode of the first city.")
    cityB: str = Field(..., description="The zipcode of the second city.")


class EstimateDriveFeasibilityByMileageInput(BaseModel):
    """Input schema for estimate_drive_feasibility_by_mileage."""
    distance: float = Field(..., description="The distance to travel in miles.")


class FillFuelTankInput(BaseModel):
    """Input schema for fillFuelTank."""
    fuelAmount: float = Field(..., description="The amount of fuel to fill in gallons; this is the additional fuel to add to the tank.")


class GallonToLiterInput(BaseModel):
    """Input schema for gallon_to_liter."""
    gallon: float = Field(..., description="The amount of gallon to convert.")


class GetZipcodeBasedOnCityInput(BaseModel):
    """Input schema for get_zipcode_based_on_city."""
    city: str = Field(..., description="The name of the city.")


class LiterToGallonInput(BaseModel):
    """Input schema for liter_to_gallon."""
    liter: float = Field(..., description="The amount of liter to convert.")


class LockDoorsInput(BaseModel):
    """Input schema for lockDoors."""
    unlock: bool = Field(..., description="True if the doors are to be unlocked, False otherwise.")
    door: List[Literal["driver", "passenger", "rear_left", "rear_right"]] = Field(..., description="The list of doors to lock or unlock.")


class PressBrakePedalInput(BaseModel):
    """Input schema for pressBrakePedal."""
    pedalPosition: float = Field(..., description="Position of the brake pedal, between 0 (not pressed) and 1 (fully pressed).")


class SetCruiseControlInput(BaseModel):
    """Input schema for setCruiseControl."""
    speed: float = Field(..., description="The speed to set in m/h. The speed should be between 0 and 120 and a multiple of 5.")
    activate: bool = Field(..., description="True to activate the cruise control, False to deactivate.")
    distanceToNextVehicle: float = Field(..., description="The distance to the next vehicle in meters.")


class SetHeadlightsInput(BaseModel):
    """Input schema for setHeadlights."""
    mode: Literal["on", "off", "auto"] = Field(..., description="The mode of the headlights.")


class SetNavigationInput(BaseModel):
    """Input schema for set_navigation."""
    destination: str = Field(..., description="The destination to navigate in the format of street, city, state.")


class StartEngineInput(BaseModel):
    """Input schema for startEngine."""
    ignitionMode: Literal["START", "STOP"] = Field(..., description="The ignition mode of the vehicle.")

