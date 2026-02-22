import asyncio
from alpaca.data.requests import StockQuotesRequest, StockTradesRequest, StockBarsRequest, StockLatestQuoteRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading import OrderSide
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import TimeInForce, OrderType, QueryOrderStatus, OrderStatus, OrderClass
import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import time

ny_timezone = ZoneInfo("America/New_York")
# market closes: 1/1, third mon in jan, third mon in feb, 
stock_list = ["APPL", "TSLA", "GOOG", "NVDA", "AMZN", "META", "MSFT"]  # change to fill this from other sources later

# Check account data
# Fill keys with respective paper trading account
API_KEY = "PKOATS2ITDVL3KE5R2PWQXCL2E"
SECRET_KEY = "Hz9S5MB5vpRHoz68CS1tQe3fWLJMokR5oN6sUn9pTJRw"
trading_client = TradingClient(API_KEY, SECRET_KEY)
account = trading_client.get_account()

print(account)
print(account.account_number)
print(account.buying_power)
print(account.cash)

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


def prev():
    # Getting quote data
    print("Test quotes")

    # Commented out is for getting requests based on time, not latest
    '''request_params = StockQuotesRequest(
        symbol_or_symbols=stock_list#,
        #start=datetime(2025, 10, 24, 9, 30),
        #end=datetime(2025, 10, 24, 9, 31),
    )
    print(request_params)
    quotes = data_client.get_stock_quotes(request_params)'''

    latest_quote_params = StockLatestQuoteRequest(
        symbol_or_symbols=stock_list
    )

    # Note to self: This gets a dictionary with ticker as keys and Quote objects as values, can save this later
    latest_quotes = data_client.get_stock_latest_quote(latest_quote_params)

    print(latest_quotes)
    print()

    # Getting Trades data
    print("Test trades data")

    latest_trade_params = StockLatestTradeRequest(
        symbol_or_symbols=stock_list
    )

    # Note to self: This gets a dictionary with ticker as keys and Quote objects as values, can save this later
    latest_trades = data_client.get_stock_latest_trade(latest_trade_params)
    print(latest_trades)
    print()
    # Test trades 1
    print("Test trades 1")
    while (float(account.cash) > 0):
        print(account.cash)
        break

    # Test trades 2
    print("Test trades 2")

    '''for stock in stock_list:
        print(stock)
        allocated_cash = round(float(account.cash)/(len(stock_list) * 1.2))
        print(allocated_cash, "dollars allocated")
        limit_price = round(latest_trades[stock].price)  # (latest_quotes[stock].ask_price + latest_quotes[stock].bid_price)/2
        if limit_price == 0:
            continue
        print(limit_price, "dollars limit price")
        buy_amount = allocated_cash//limit_price
        print(buy_amount, "shares")
    
        # Save this as some point
        limit_order_data = LimitOrderRequest(
            symbol=stock,
            qty=buy_amount,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.FOK,  # Currently fill or kill, used to be DAY
            limit_price=limit_price,
            extended_hours=True
        )
    
        limit_order = trading_client.submit_order(limit_order_data)
    
    order_params = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
    )
    positions = trading_client.get_all_positions()
    print("Positions:")
    print(positions)
    time.sleep(1.5)
    
    # Set stop loss/take profit
    for position in positions:
        take_profit_data = TakeProfitRequest(
            limit_price=round(float(position.avg_entry_price) * 1.02, 2)
        )
        stop_loss_data = StopLossRequest(
            stop_price=round(float(position.avg_entry_price) * 0.98, 2)
            # You could also add a limit_price here to make it a stop-limit
            # limit_price=144.90
        )
        try:
            bracket_order_data = MarketOrderRequest(
                symbol=position.symbol,
                qty=position.qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,  # Good 'Til Canceled (common for brackets)
                order_class=OrderClass.BRACKET,  # This makes it a bracket order
                take_profit=take_profit_data,  # Attach the take-profit
                stop_loss=stop_loss_data  # Attach the stop-loss
            )
    
            # --- 4. Submit the Order ---
            submitted_order = trading_client.submit_order(order_data=bracket_order_data)
            print(f"Successfully submitted BRACKET order for {submitted_order.symbol}.")
            print(f"  Order ID: {submitted_order.id}")
    
        except Exception as e:
            print(f"An error occurred submitting the bracket order: {e}")'''

    try:
        cash_per_stock = float(account.cash) / (len(stock_list) * 1.2)
    except ZeroDivisionError:
        print("Error: stock_list is empty or account.cash is 0.")
        cash_per_stock = 0

    print(f"Allocating ${cash_per_stock:.2f} per stock...")
    submitted_orders = []

    for stock in stock_list:
        try:
            limit_price = round(latest_trades[stock].price, 2)  # Round to 2 decimals
            if limit_price == 0:
                print(f"Skipping {stock}: Invalid limit price $0.")
                continue

            buy_amount = int(cash_per_stock // limit_price)  # Use int() for whole shares
            if buy_amount <= 0:
                print(f"Skipping {stock}: Not enough cash to buy 1 share at ${limit_price}.")
                continue

            print(f"Preparing order for {buy_amount} shares of {stock} at ${limit_price}...")

            # --- Define exit legs *before* submitting ---
            take_profit_data = TakeProfitRequest(
                limit_price=round(limit_price * 1.02, 2)  # 2% profit
            )
            stop_loss_data = StopLossRequest(
                stop_price=round(limit_price * 0.98, 2)  # 2% stop
            )

            # --- Build the *initial* order as a Bracket Order ---
            limit_order_data = LimitOrderRequest(
                symbol=stock,
                qty=buy_amount,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,  # Use DAY to let the order rest
                limit_price=limit_price,
                order_class=OrderClass.BRACKET,  # This is the key
                take_profit=take_profit_data,
                stop_loss=stop_loss_data
            )

            # --- Submit the single bracket order ---
            limit_order = trading_client.submit_order(limit_order_data)
            submitted_orders.append(limit_order)
            print(f"Successfully submitted BRACKET limit order for {stock}.")
            print(f"  Order ID: {limit_order.id}")

        except Exception as e:
            print(f"Error processing {stock}: {e}")

    print("\n--- All orders submitted ---")
    for order in submitted_orders:
        print(f"ID: {order.id}, Symbol: {order.symbol}, Status: {order.status}")

    # Stream then test sell
    '''stream = StockDataStream("PKOATS2ITDVL3KE5R2PWQXCL2E", "Hz9S5MB5vpRHoz68CS1tQe3fWLJMokR5oN6sUn9pTJRw")
    
    async def handle_stream(data):
        print(data)
        positions = trading_client.get_all_positions()
        if positions == {}:
            print("No open position, stopping stream")
            asyncio.create_task(stream.stop_ws())
        if data.timestamp.hour >= 16:
            print("Time's up, closing all positions")
            asyncio.create_task(stream.stop_ws())
    
        if data.price >= round(latest_trades[data.symbol].price * 1.02, 2) or data.price >= round(latest_trades[data.symbol].price * 0.98, 2):
            trading_client.close_position(data.symbol)
    
    
    stream.subscribe_trades(handle_stream, *positions)
    stream.run()'''


def get_historical_data(symbol):
    end_time = datetime.datetime.now()

    start_time = end_time - datetime.timedelta(days=100)

    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=start_time,
        end=end_time
    )

    bars = data_client.get_stock_bars(request_params)
    return bars.df


def calculate_rsi(data, period=14):
    close = data['close']

    delta = close.diff()

    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def execute_trade(signal, symbol):
    if signal == "buy":
        print(f"Buying {symbol}...")
        market_order = MarketOrderRequest(
            symbol=symbol,
            qty=1,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order_data=market_order)

    elif signal == "sell":
        print(f"Selling {symbol}...")
        market_order = MarketOrderRequest(
            symbol=symbol,
            qty=1,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order_data=market_order)


def main():

    for stock in stock_list:
        print(f"Analyzing {stock}...")

        df = get_historical_data(stock)

        df['RSI'] = calculate_rsi(df)

        current_rsi = df['RSI'].iloc[-1]
        print(f"Current RSI: {current_rsi:.2f}")

        if current_rsi < 30:
            print("Signal: OVERSOLD (Buy Opportunity)")
            # execute_trade("buy", stock)  # Uncomment to trade
        elif current_rsi > 70:
            print("Signal: OVERBOUGHT (Sell Opportunity)")
            # execute_trade("sell", stock) # Uncomment to trade
        else:
            print("Signal: NEUTRAL (No Action)")


if __name__ == "__main__":
    main()
