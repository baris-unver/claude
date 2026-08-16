//+------------------------------------------------------------------+
//|                                          XAUUSD_Sequential_EA.mq5 |
//|  Sequential short/long XAUUSD strategy for Strategy Tester        |
//|                                                                    |
//|  Implements the rule set strictly and in order:                    |
//|   1    open short 0.01 at market, entry price = x                  |
//|   2..2-i trail the short's SL through fixed bands of x             |
//|   2-j  close the short below 0.9775x                               |
//|   3    open long 0.03 at CP >= 1.001x, entry price = y             |
//|   4..4-h once per minute: if 2.25 <= (CP-x)/(y-x) < 12,            |
//|        set long SL = CP - 1 USD                                    |
//|   4-i  close both at ratio >= 12                                   |
//|   4-j  if the long's SL fires, close the short too                 |
//|   5    with both open and CP < x, open short 0.02, entry = z       |
//|   6    close all three when CP < 0.998x (z >= 0.9995x)             |
//|        or CP < 0.9975x (z < 0.9995x)                               |
//|                                                                    |
//|  Requires a hedging account (three concurrent positions on one     |
//|  symbol). CP is the Bid price throughout.                          |
//+------------------------------------------------------------------+
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

input double InpLotShort1       = 0.01;      // Stage 1 short lot
input double InpLotLong         = 0.03;      // Stage 3 long lot
input double InpLotShort3       = 0.02;      // Stage 5 short lot
input double InpLongSlOffsetUsd = 1.0;       // Stage 4 SL distance below CP (USD)
input bool   InpRestartAfterCycle = true;    // Start a new cycle after all positions close
input long   InpMagic           = 20260816;  // Magic number

enum EState
  {
   ST_OPEN_SHORT = 0,   // stage 1
   ST_MANAGE_SHORT,     // stages 2 / 3
   ST_MANAGE_PAIR,      // stages 4 / 5
   ST_MANAGE_THREE,     // stage 6
   ST_DONE
  };

CTrade   trade;
EState   state    = ST_OPEN_SHORT;
ulong    tkShort1 = 0, tkLong = 0, tkShort3 = 0;
double   x = 0.0, y = 0.0, z = 0.0;
datetime lastM1   = 0;

//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetTypeFillingBySymbol(_Symbol);
   trade.SetDeviationInPoints(50);

   if((ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE)
      != ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
      Print("WARNING: netting account detected - the strategy needs hedging ",
            "to hold a short and a long on ", _Symbol, " simultaneously.");
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
bool PosExists(const ulong tk)
  {
   return tk != 0 && PositionSelectByTicket(tk);
  }

//+------------------------------------------------------------------+
bool OpenMarket(const bool isSell, const double lots, ulong &ticket, double &fill)
  {
   bool ok = isSell ? trade.Sell(lots, _Symbol) : trade.Buy(lots, _Symbol);
   if(!ok)
     {
      Print("Open failed: ", trade.ResultRetcodeDescription());
      return false;
     }
   ulong deal = trade.ResultDeal();
   if(deal > 0 && HistoryDealSelect(deal))
     {
      ticket = (ulong)HistoryDealGetInteger(deal, DEAL_POSITION_ID);
      fill   = HistoryDealGetDouble(deal, DEAL_PRICE);
     }
   else
     {
      ticket = trade.ResultOrder();
      fill   = trade.ResultPrice();
     }
   if(fill <= 0.0 && PositionSelectByTicket(ticket))
      fill = PositionGetDouble(POSITION_PRICE_OPEN);
   return ticket > 0 && fill > 0.0;
  }

//+------------------------------------------------------------------+
void CycleEnd(const string reason)
  {
   PrintFormat("Cycle finished (%s). x=%.2f y=%.2f z=%.2f", reason, x, y, z);
   tkShort1 = tkLong = tkShort3 = 0;
   x = y = z = 0.0;
   state = InpRestartAfterCycle ? ST_OPEN_SHORT : ST_DONE;
  }

//+------------------------------------------------------------------+
//| Stages 2 and 3: trail the short's SL, watch for the long trigger |
//+------------------------------------------------------------------+
void ManageShortStage(const double cp)
  {
   // the trailing SL fired: every stage-2 SL is below x, so profit is locked
   if(!PosExists(tkShort1))
     {
      CycleEnd("stage-2 trailing SL hit");
      return;
     }

   // 2-j
   if(cp <= 0.9775 * x)
     {
      trade.PositionClose(tkShort1);
      CycleEnd("rule 2-j close");
      return;
     }

   // 3
   if(cp >= 1.001 * x)
     {
      if(OpenMarket(false, InpLotLong, tkLong, y))
        {
         lastM1 = iTime(_Symbol, PERIOD_M1, 0);
         state  = ST_MANAGE_PAIR;
         PrintFormat("Stage 3: long %.2f lots at y=%.2f (x=%.2f)", InpLotLong, y, x);
        }
      return;
     }

   // 2 .. 2-i
   double slFrac = 0.0;
   if(cp < 0.999 * x)
     {
      if(cp > 0.9985 * x)      slFrac = 0.99925; // 2
      else if(cp > 0.998  * x) slFrac = 0.99875; // 2-a
      else if(cp > 0.9975 * x) slFrac = 0.99825; // 2-b
      else if(cp > 0.997  * x) slFrac = 0.99775; // 2-c
      else if(cp > 0.9955 * x) slFrac = 0.99725; // 2-d
      else if(cp > 0.994  * x) slFrac = 0.99575; // 2-e
      else if(cp > 0.9925 * x) slFrac = 0.99425; // 2-f
      else if(cp > 0.99   * x) slFrac = 0.993;   // 2-g
      else if(cp > 0.985  * x) slFrac = 0.9905;  // 2-h
      else                     slFrac = 0.9855;  // 2-i  (cp > 0.9775x here)
     }
   if(slFrac > 0.0)
     {
      double sl  = NormalizeDouble(slFrac * x, _Digits);
      double cur = PositionGetDouble(POSITION_SL);
      if(MathAbs(cur - sl) > _Point / 2.0)
         if(!trade.PositionModify(tkShort1, sl, 0.0))
            Print("Short SL modify failed: ", trade.ResultRetcodeDescription());
     }
  }

//+------------------------------------------------------------------+
//| Stages 4 and 5: minute-based long trailing, third-position entry |
//+------------------------------------------------------------------+
void ManagePairStage(const double cp)
  {
   bool longOpen  = PosExists(tkLong);
   bool shortOpen = PosExists(tkShort1);

   // 4-j: the long's SL (set at CP-1 above breakeven territory) fired
   if(!longOpen)
     {
      if(shortOpen)
         trade.PositionClose(tkShort1);
      CycleEnd("rule 4-j: long SL hit, short closed");
      return;
     }
   if(!shortOpen) // not reachable by the rules; safety net
     {
      trade.PositionClose(tkLong);
      CycleEnd("short disappeared unexpectedly");
      return;
     }

   // 5: both open and price back below x -> third position
   if(cp < x)
     {
      if(OpenMarket(true, InpLotShort3, tkShort3, z))
        {
         state = ST_MANAGE_THREE;
         PrintFormat("Stage 5: short %.2f lots at z=%.2f (x=%.2f)", InpLotShort3, z, x);
        }
      return;
     }

   // 4 .. 4-i are evaluated once per minute ("wait till the next minute value")
   datetime m1 = iTime(_Symbol, PERIOD_M1, 0);
   if(m1 == lastM1)
      return;
   lastM1 = m1;

   double denom = y - x;
   if(denom <= 0.0)
      return;
   double r = (cp - x) / denom;

   if(r >= 12.0) // 4-i
     {
      trade.PositionClose(tkLong);
      trade.PositionClose(tkShort1);
      CycleEnd("rule 4-i: ratio >= 12");
      return;
     }
   if(r >= 2.25) // 4 .. 4-h all set the same SL
     {
      double sl = NormalizeDouble(cp - InpLongSlOffsetUsd, _Digits);
      PositionSelectByTicket(tkLong);
      double cur = PositionGetDouble(POSITION_SL);
      if(MathAbs(cur - sl) > _Point / 2.0)
         if(!trade.PositionModify(tkLong, sl, 0.0))
            Print("Long SL modify failed: ", trade.ResultRetcodeDescription());
     }
  }

//+------------------------------------------------------------------+
//| Stage 6: exit all three below the z-dependent threshold          |
//+------------------------------------------------------------------+
void ManageThreeStage(const double cp)
  {
   double threshold = (z >= 0.9995 * x) ? 0.998 * x : 0.9975 * x;
   if(cp < threshold)
     {
      if(PosExists(tkShort1)) trade.PositionClose(tkShort1);
      if(PosExists(tkLong))   trade.PositionClose(tkLong);
      if(PosExists(tkShort3)) trade.PositionClose(tkShort3);
      CycleEnd("rule 6 close-all");
      return;
     }
   if(!PosExists(tkShort1) && !PosExists(tkLong) && !PosExists(tkShort3))
      CycleEnd("all positions gone in stage 6"); // safety net
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   double cp = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   switch(state)
     {
      case ST_OPEN_SHORT:
         if(OpenMarket(true, InpLotShort1, tkShort1, x))
           {
            state = ST_MANAGE_SHORT;
            PrintFormat("Stage 1: short %.2f lots at x=%.2f", InpLotShort1, x);
           }
         break;
      case ST_MANAGE_SHORT: ManageShortStage(cp); break;
      case ST_MANAGE_PAIR:  ManagePairStage(cp);  break;
      case ST_MANAGE_THREE: ManageThreeStage(cp); break;
      case ST_DONE:         break;
     }
  }
//+------------------------------------------------------------------+
