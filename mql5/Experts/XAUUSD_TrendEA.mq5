//+------------------------------------------------------------------+
//|                                              XAUUSD_TrendEA.mq5  |
//|  XAUUSD trend-following Expert Advisor                           |
//|                                                                  |
//|  Strategy (evaluated on closed bars, orders sent on new bar):    |
//|    Trend filter : EMA(fast) vs EMA(slow)                         |
//|    Entry        : close crosses back over EMA(fast) in the       |
//|                   direction of the trend, confirmed by RSI       |
//|    Stops        : ATR-based SL/TP, optional ATR trailing stop    |
//|    Sizing       : fixed risk % of balance per trade              |
//|    Filters      : trading session hours, max spread,             |
//|                   minimum ATR, daily loss limit                  |
//|                                                                  |
//|  The Python harness in /backtest implements the same rules so    |
//|  results can be cross-checked outside MetaTrader.                |
//+------------------------------------------------------------------+
#property copyright "baris-unver"
#property link      "https://github.com/baris-unver"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

//--- inputs: strategy core
input group    "=== Strategy ==="
input ENUM_TIMEFRAMES InpTimeframe      = PERIOD_H1;  // Signal timeframe
input int      InpEmaFastPeriod         = 50;         // EMA fast period
input int      InpEmaSlowPeriod         = 200;        // EMA slow period
input int      InpRsiPeriod             = 14;         // RSI period
input double   InpRsiLongMin            = 52.0;       // RSI minimum for longs
input double   InpRsiShortMax           = 48.0;       // RSI maximum for shorts
input int      InpAtrPeriod             = 14;         // ATR period
input double   InpAtrMin                = 0.0;        // Min ATR in price units (0 = off)
input bool     InpAllowLongs            = true;       // Allow long trades
input bool     InpAllowShorts           = true;       // Allow short trades
input bool     InpCloseOnOpposite       = true;       // Close position on opposite signal

input group    "=== Risk & exits ==="
input double   InpRiskPercent           = 1.0;        // Risk per trade (% of balance)
input double   InpSlAtrMult             = 2.0;        // Stop-loss distance (x ATR)
input double   InpTpAtrMult             = 3.0;        // Take-profit distance (x ATR)
input double   InpTrailAtrMult          = 0.0;        // Trailing stop (x ATR, 0 = off)
input double   InpDailyLossLimitPct     = 0.0;        // Daily loss limit (% of day-start balance, 0 = off)

input group    "=== Filters ==="
input bool     InpUseSessionFilter      = true;       // Use session filter (server time)
input int      InpSessionStartHour      = 7;          // Session start hour (inclusive)
input int      InpSessionEndHour        = 20;         // Session end hour (exclusive)
input long     InpMaxSpreadPoints       = 50;         // Max spread in points (0 = off)

input group    "=== Housekeeping ==="
input long     InpMagicNumber           = 20260816;   // Magic number
input string   InpTradeComment          = "XAUUSD_TrendEA";

//--- globals
CTrade   g_trade;
int      g_emaFastHandle = INVALID_HANDLE;
int      g_emaSlowHandle = INVALID_HANDLE;
int      g_rsiHandle     = INVALID_HANDLE;
int      g_atrHandle     = INVALID_HANDLE;
datetime g_lastBarTime   = 0;
datetime g_currentDay    = 0;
double   g_dayStartBalance = 0.0;

//+------------------------------------------------------------------+
//| Initialization                                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   if(InpEmaFastPeriod >= InpEmaSlowPeriod)
     {
      Print("Init error: EMA fast period must be smaller than EMA slow period");
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(InpSlAtrMult <= 0.0 || InpRiskPercent <= 0.0)
     {
      Print("Init error: SL ATR multiple and risk percent must be positive");
      return(INIT_PARAMETERS_INCORRECT);
     }

   g_emaFastHandle = iMA(_Symbol, InpTimeframe, InpEmaFastPeriod, 0, MODE_EMA, PRICE_CLOSE);
   g_emaSlowHandle = iMA(_Symbol, InpTimeframe, InpEmaSlowPeriod, 0, MODE_EMA, PRICE_CLOSE);
   g_rsiHandle     = iRSI(_Symbol, InpTimeframe, InpRsiPeriod, PRICE_CLOSE);
   g_atrHandle     = iATR(_Symbol, InpTimeframe, InpAtrPeriod);

   if(g_emaFastHandle == INVALID_HANDLE || g_emaSlowHandle == INVALID_HANDLE ||
      g_rsiHandle == INVALID_HANDLE || g_atrHandle == INVALID_HANDLE)
     {
      Print("Init error: failed to create indicator handles");
      return(INIT_FAILED);
     }

   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(20);
   g_trade.SetTypeFillingBySymbol(_Symbol);

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Deinitialization                                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(g_emaFastHandle != INVALID_HANDLE) IndicatorRelease(g_emaFastHandle);
   if(g_emaSlowHandle != INVALID_HANDLE) IndicatorRelease(g_emaSlowHandle);
   if(g_rsiHandle     != INVALID_HANDLE) IndicatorRelease(g_rsiHandle);
   if(g_atrHandle     != INVALID_HANDLE) IndicatorRelease(g_atrHandle);
  }

//+------------------------------------------------------------------+
//| Main tick handler                                                |
//+------------------------------------------------------------------+
void OnTick()
  {
   UpdateDailyAnchor();

   if(InpTrailAtrMult > 0.0)
      ManageTrailingStop();

   if(!IsNewBar())
      return;

   //--- read indicator values for the two most recent closed bars
   double emaFast[3], emaSlow[3], rsi[3], atr[3];
   if(CopyBuffer(g_emaFastHandle, 0, 0, 3, emaFast) < 3) return;
   if(CopyBuffer(g_emaSlowHandle, 0, 0, 3, emaSlow) < 3) return;
   if(CopyBuffer(g_rsiHandle,     0, 0, 3, rsi)     < 3) return;
   if(CopyBuffer(g_atrHandle,     0, 0, 3, atr)     < 3) return;
   // buffers are ordered oldest->newest here; index 2 = current forming bar,
   // index 1 = signal bar (last closed), index 0 = bar before the signal bar

   double close1 = iClose(_Symbol, InpTimeframe, 1);   // signal bar close
   double close2 = iClose(_Symbol, InpTimeframe, 2);   // prior bar close
   double emaFast1 = emaFast[1], emaFast0 = emaFast[0];
   double emaSlow1 = emaSlow[1];
   double rsi1 = rsi[1];
   double atr1 = atr[1];

   if(atr1 <= 0.0)
      return;

   //--- signal logic (must stay in sync with backtest/strategy.py)
   bool trendUp   = emaFast1 > emaSlow1;
   bool trendDown = emaFast1 < emaSlow1;
   bool crossUp   = (close2 <= emaFast0) && (close1 > emaFast1);
   bool crossDown = (close2 >= emaFast0) && (close1 < emaFast1);

   bool longSignal  = InpAllowLongs  && trendUp   && crossUp   && (rsi1 >= InpRsiLongMin);
   bool shortSignal = InpAllowShorts && trendDown && crossDown && (rsi1 <= InpRsiShortMax);

   //--- opposite-signal exit acts even outside the session window
   if(InpCloseOnOpposite && PositionSelect(_Symbol))
     {
      long posType = PositionGetInteger(POSITION_TYPE);
      long posMagic = PositionGetInteger(POSITION_MAGIC);
      if(posMagic == InpMagicNumber)
        {
         if(posType == POSITION_TYPE_BUY && shortSignal)
            g_trade.PositionClose(_Symbol);
         else if(posType == POSITION_TYPE_SELL && longSignal)
            g_trade.PositionClose(_Symbol);
        }
     }

   //--- entry filters
   if(!longSignal && !shortSignal)
      return;
   if(PositionSelect(_Symbol))                 // one position at a time
      return;
   if(InpAtrMin > 0.0 && atr1 < InpAtrMin)
      return;
   if(!SessionAllows())
      return;
   if(!SpreadAllows())
      return;
   if(!DailyLossAllows())
      return;

   //--- place order
   double slDistance = InpSlAtrMult * atr1;
   double tpDistance = InpTpAtrMult * atr1;
   double lots = CalcLots(slDistance);
   if(lots <= 0.0)
      return;

   if(longSignal)
     {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl  = NormalizeDouble(ask - slDistance, _Digits);
      double tp  = (InpTpAtrMult > 0.0) ? NormalizeDouble(ask + tpDistance, _Digits) : 0.0;
      g_trade.Buy(lots, _Symbol, 0.0, sl, tp, InpTradeComment);
     }
   else if(shortSignal)
     {
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double sl  = NormalizeDouble(bid + slDistance, _Digits);
      double tp  = (InpTpAtrMult > 0.0) ? NormalizeDouble(bid - tpDistance, _Digits) : 0.0;
      g_trade.Sell(lots, _Symbol, 0.0, sl, tp, InpTradeComment);
     }
  }

//+------------------------------------------------------------------+
//| True once per new bar of the signal timeframe                    |
//+------------------------------------------------------------------+
bool IsNewBar()
  {
   datetime barTime = iTime(_Symbol, InpTimeframe, 0);
   if(barTime == g_lastBarTime)
      return(false);
   g_lastBarTime = barTime;
   return(true);
  }

//+------------------------------------------------------------------+
//| Track day-start balance for the daily loss limit                 |
//+------------------------------------------------------------------+
void UpdateDailyAnchor()
  {
   datetime now = TimeCurrent();
   MqlDateTime dt;
   TimeToStruct(now, dt);
   dt.hour = 0; dt.min = 0; dt.sec = 0;
   datetime day = StructToTime(dt);
   if(day != g_currentDay)
     {
      g_currentDay = day;
      g_dayStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
     }
  }

//+------------------------------------------------------------------+
//| Session filter (server time)                                     |
//+------------------------------------------------------------------+
bool SessionAllows()
  {
   if(!InpUseSessionFilter)
      return(true);
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   if(InpSessionStartHour <= InpSessionEndHour)
      return(dt.hour >= InpSessionStartHour && dt.hour < InpSessionEndHour);
   // overnight session (e.g. 22 -> 6)
   return(dt.hour >= InpSessionStartHour || dt.hour < InpSessionEndHour);
  }

//+------------------------------------------------------------------+
//| Spread filter                                                    |
//+------------------------------------------------------------------+
bool SpreadAllows()
  {
   if(InpMaxSpreadPoints <= 0)
      return(true);
   long spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   return(spread <= InpMaxSpreadPoints);
  }

//+------------------------------------------------------------------+
//| Daily loss limit                                                 |
//+------------------------------------------------------------------+
bool DailyLossAllows()
  {
   if(InpDailyLossLimitPct <= 0.0 || g_dayStartBalance <= 0.0)
      return(true);
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double lossPct = (g_dayStartBalance - balance) / g_dayStartBalance * 100.0;
   return(lossPct < InpDailyLossLimitPct);
  }

//+------------------------------------------------------------------+
//| Risk-based position sizing                                       |
//+------------------------------------------------------------------+
double CalcLots(double slDistance)
  {
   if(slDistance <= 0.0)
      return(0.0);

   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   if(tickSize <= 0.0 || tickValue <= 0.0)
      return(0.0);

   double riskMoney       = AccountInfoDouble(ACCOUNT_BALANCE) * InpRiskPercent / 100.0;
   double lossPerLot      = slDistance / tickSize * tickValue;  // money lost per lot if SL hit
   if(lossPerLot <= 0.0)
      return(0.0);
   double lots = riskMoney / lossPerLot;

   double minLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(lotStep > 0.0)
      lots = MathFloor(lots / lotStep) * lotStep;
   if(lots < minLot)
      return(0.0);   // refuse to trade oversized risk rather than bump up to min lot
   if(lots > maxLot)
      lots = maxLot;
   return(NormalizeDouble(lots, 2));
  }

//+------------------------------------------------------------------+
//| ATR trailing stop (tightens only)                                |
//+------------------------------------------------------------------+
void ManageTrailingStop()
  {
   if(!PositionSelect(_Symbol))
      return;
   if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
      return;

   double atr[1];
   if(CopyBuffer(g_atrHandle, 0, 1, 1, atr) < 1)   // last closed bar's ATR
      return;
   double trailDistance = InpTrailAtrMult * atr[0];
   if(trailDistance <= 0.0)
      return;

   long   posType   = PositionGetInteger(POSITION_TYPE);
   double currentSl = PositionGetDouble(POSITION_SL);
   double currentTp = PositionGetDouble(POSITION_TP);

   if(posType == POSITION_TYPE_BUY)
     {
      double bid   = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double newSl = NormalizeDouble(bid - trailDistance, _Digits);
      if(newSl > currentSl && newSl < bid)
         g_trade.PositionModify(_Symbol, newSl, currentTp);
     }
   else if(posType == POSITION_TYPE_SELL)
     {
      double ask   = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double newSl = NormalizeDouble(ask + trailDistance, _Digits);
      if((currentSl == 0.0 || newSl < currentSl) && newSl > ask)
         g_trade.PositionModify(_Symbol, newSl, currentTp);
     }
  }
//+------------------------------------------------------------------+
