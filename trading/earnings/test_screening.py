"""Standalone test script for ML + LLM screening pipeline."""

import argparse
import asyncio
import logging
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description='Test earnings screening pipeline')
    parser.add_argument('--ticker', '-t', nargs='+', help='Specific ticker(s) to screen')
    parser.add_argument('--earnings-date', '-d', type=str, help='Earnings date (YYYY-MM-DD)')
    parser.add_argument('--timing', choices=['BMO', 'AMC'], default='AMC', help='Earnings timing')
    parser.add_argument('--days-ahead', type=int, default=3, help='Days ahead to fetch earnings')
    parser.add_argument('--spread-threshold', type=float, default=15.0, help='Max spread %%')
    parser.add_argument('--edge-threshold', type=float, default=0.05, help='Min edge (0.05 = 5%%)')
    parser.add_argument('--no-ibkr', action='store_true', help='Skip IBKR screening')
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM sanity check')
    parser.add_argument('--ibkr-port', type=int, default=4002, help='IBKR Gateway port')
    args = parser.parse_args()

    # Step 1: Get earnings events
    if args.ticker and args.earnings_date:
        # Manual ticker + date
        from trading.earnings.screener import EarningsEvent
        earnings_date = datetime.strptime(args.earnings_date, '%Y-%m-%d').date()
        events = [EarningsEvent(symbol=t, earnings_date=earnings_date, timing=args.timing) for t in args.ticker]
    else:
        # Fetch from Nasdaq
        from trading.earnings.screener import fetch_upcoming_earnings
        events = fetch_upcoming_earnings(days_ahead=args.days_ahead)
        if args.ticker:
            events = [e for e in events if e.symbol in args.ticker]

    logger.info(f"Found {len(events)} earnings events")
    for e in events[:10]:
        logger.info(f"  {e.symbol}: {e.earnings_date} ({e.timing})")
    if len(events) > 10:
        logger.info(f"  ... and {len(events) - 10} more")

    # Step 2: IBKR screening (if enabled)
    candidates = []
    if not args.no_ibkr:
        from ib_insync import IB
        from trading.earnings.screener import screen_all_candidates

        ib = IB()
        try:
            await ib.connectAsync('127.0.0.1', args.ibkr_port, clientId=99)
            logger.info("Connected to IBKR")

            passed, rejected = await screen_all_candidates(
                ib, events,
                spread_threshold=args.spread_threshold,
                max_candidates=50
            )
            candidates = passed

            logger.info(f"\nScreening: {len(passed)} passed, {len(rejected)} rejected")
            for r in rejected[:5]:
                logger.info(f"  REJECTED {r.symbol}: {r.rejection_reason}")
            if len(rejected) > 5:
                logger.info(f"  ... and {len(rejected) - 5} more rejections")
        finally:
            ib.disconnect()
    else:
        # Mock candidates for testing ML/LLM without IBKR
        from trading.earnings.screener import ScreenedCandidate
        logger.info("\nSkipping IBKR screening, using mock candidate data")
        for e in events:
            candidates.append(ScreenedCandidate(
                symbol=e.symbol,
                earnings_date=e.earnings_date,
                timing=e.timing,
                expiry="2026-01-17",  # Mock
                atm_strike=100.0,
                spot_price=100.0,
                call_bid=2.0, call_ask=2.20, call_iv=0.50,
                put_bid=2.0, put_ask=2.20, put_iv=0.50,
                straddle_mid=4.10,
                spread_pct=4.9,
                implied_move_pct=4.1,
            ))

    if not candidates:
        logger.info("No candidates passed screening")
        return

    # Step 3: ML predictions
    from trading.earnings.ml_predictor import get_predictor
    predictor = get_predictor()

    if not predictor:
        logger.error("ML predictor not available (check model files)")
        return

    logger.info(f"\n--- ML Predictions ---")
    ml_passed = []
    for c in candidates:
        prediction = predictor.predict(c.symbol, c.earnings_date, c.timing)
        if prediction:
            edge = prediction.edge_q75
            status = "PASS" if edge >= args.edge_threshold else "FAIL"
            logger.info(f"{c.symbol}: q75={prediction.pred_q75:.1%}, implied={c.implied_move_pct:.1f}%, edge={edge:.1%} -> {status}")
            if edge >= args.edge_threshold:
                c.edge_q75 = edge
                c.pred_q75 = prediction.pred_q75
                ml_passed.append((c, prediction))
        else:
            logger.warning(f"{c.symbol}: No prediction available")

    logger.info(f"\nML filter: {len(ml_passed)} passed edge threshold ({args.edge_threshold:.0%})")

    # Step 4: LLM sanity check (if enabled)
    if not args.no_llm and ml_passed:
        from trading.earnings.llm_sanity_check import build_sanity_packet, check_with_llm

        logger.info(f"\n--- LLM Sanity Checks ---")
        for candidate, prediction in ml_passed:
            packet = build_sanity_packet(candidate, prediction)
            result = await check_with_llm(packet, None, ticker=candidate.symbol)

            flags_str = ', '.join(result.risk_flags) if result.risk_flags else 'none'
            logger.info(f"{candidate.symbol}: {result.decision} (flags: {flags_str}, latency: {result.latency_ms}ms)")
            if result.reasons:
                for reason in result.reasons:
                    logger.info(f"  - {reason}")
    elif args.no_llm:
        logger.info("\n--- LLM Sanity Check skipped (--no-llm) ---")


if __name__ == '__main__':
    asyncio.run(main())
