"""Standalone test script for ML + LLM screening pipeline."""

import argparse
import asyncio
import logging
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def show_preview():
    """Show weekly earnings calendar preview (no screening)."""
    import pytz
    from trading.earnings.screener import fetch_upcoming_earnings
    
    ET = pytz.timezone('US/Eastern')
    today = datetime.now(ET).date()
    
    # Fetch 10 days to cover next week fully
    events = fetch_upcoming_earnings(days_ahead=10)
    
    # Group by date
    by_date = {}
    for e in events:
        if e.earnings_date not in by_date:
            by_date[e.earnings_date] = {'BMO': [], 'AMC': [], 'unknown': []}
        timing_key = e.timing if e.timing in ('BMO', 'AMC') else 'unknown'
        by_date[e.earnings_date][timing_key].append(e.symbol)
    
    # Print calendar
    print(f"\n{'='*60}")
    print(f"EARNINGS CALENDAR: {today} to {today + timedelta(days=10)}")
    print(f"{'='*60}\n")
    
    for d in sorted(by_date.keys()):
        if d < today:
            continue
        
        day_name = d.strftime('%A')
        bmo = by_date[d]['BMO']
        amc = by_date[d]['AMC']
        unknown = by_date[d]['unknown']
        total = len(bmo) + len(amc) + len(unknown)
        
        # Mark today
        marker = " (TODAY)" if d == today else ""
        print(f"=== {day_name} {d}{marker} ({total} total) ===")
        
        if bmo:
            symbols = ' '.join(bmo[:15])
            more = f"... +{len(bmo)-15} more" if len(bmo) > 15 else ""
            print(f"  BMO ({len(bmo):2d}): {symbols}{more}")
        if amc:
            symbols = ' '.join(amc[:15])
            more = f"... +{len(amc)-15} more" if len(amc) > 15 else ""
            print(f"  AMC ({len(amc):2d}): {symbols}{more}")
        if unknown:
            symbols = ' '.join(unknown[:10])
            more = f"... +{len(unknown)-10} more" if len(unknown) > 10 else ""
            print(f"  TBD ({len(unknown):2d}): {symbols}{more}")
        print()
    
    # Summary
    total_events = sum(len(bmo) + len(amc) + len(unknown) 
                       for bmo, amc, unknown in [(by_date[d]['BMO'], by_date[d]['AMC'], by_date[d]['unknown']) 
                                                  for d in by_date if d >= today])
    print(f"Total: {total_events} earnings events in next 10 days")


async def main():
    parser = argparse.ArgumentParser(description='Test earnings screening pipeline')
    parser.add_argument('--ticker', '-t', nargs='+', help='Specific ticker(s) to screen')
    parser.add_argument('--earnings-date', '-d', type=str, help='Earnings date (YYYY-MM-DD)')
    parser.add_argument('--timing', choices=['BMO', 'AMC'], default='AMC', help='Earnings timing')
    parser.add_argument('--preview', action='store_true', help='Show weekly earnings calendar (no screening)')
    parser.add_argument('--spread-threshold', type=float, default=15.0, help='Max spread %%')
    parser.add_argument('--edge-threshold', type=float, default=0.05, help='Min edge (0.05 = 5%%)')
    parser.add_argument('--no-ibkr', action='store_true', help='Skip IBKR screening')
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM sanity check')
    parser.add_argument('--ibkr-port', type=int, default=4002, help='IBKR Gateway port')
    args = parser.parse_args()
    
    # Preview mode: just show calendar and exit
    if args.preview:
        show_preview()
        return

    # Step 1: Get earnings events
    if args.ticker and args.earnings_date:
        # Manual ticker + date
        from trading.earnings.screener import EarningsEvent
        earnings_date = datetime.strptime(args.earnings_date, '%Y-%m-%d').date()
        events = [EarningsEvent(symbol=t, earnings_date=earnings_date, timing=args.timing) for t in args.ticker]
    else:
        # Fetch tradeable candidates: Today AMC + Tomorrow BMO
        from trading.earnings.screener import get_tradeable_candidates
        import pytz
        ET = pytz.timezone('US/Eastern')
        today = datetime.now(ET).date()
        
        bmo_tomorrow, amc_today = get_tradeable_candidates(
            fill_timing=True,
            verify_dates=False,  # Skip FMP verification for faster testing
        )
        
        # Combine: we trade both today AMC and tomorrow BMO in same session
        events = amc_today + bmo_tomorrow
        
        if args.ticker:
            events = [e for e in events if e.symbol in args.ticker]

    logger.info(f"Found {len(events)} tradeable earnings events")
    
    # Show breakdown by timing
    from datetime import timedelta
    import pytz
    ET = pytz.timezone('US/Eastern')
    today = datetime.now(ET).date()
    tomorrow = today + timedelta(days=1)
    
    amc_today_syms = [e for e in events if e.earnings_date == today and e.timing == 'AMC']
    bmo_tomorrow_syms = [e for e in events if e.earnings_date == tomorrow and e.timing == 'BMO']
    
    if amc_today_syms:
        logger.info(f"  Today AMC ({today}): {', '.join(e.symbol for e in amc_today_syms[:10])}")
        if len(amc_today_syms) > 10:
            logger.info(f"    ... and {len(amc_today_syms) - 10} more")
    if bmo_tomorrow_syms:
        logger.info(f"  Tomorrow BMO ({tomorrow}): {', '.join(e.symbol for e in bmo_tomorrow_syms[:10])}")
        if len(bmo_tomorrow_syms) > 10:
            logger.info(f"    ... and {len(bmo_tomorrow_syms) - 10} more")

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
            for r in rejected:
                logger.info(f"  REJECTED {r.symbol}: {r.rejection_reason}")
        finally:
            ib.disconnect()
    else:
        # Mock candidates for testing ML/LLM without IBKR
        from trading.earnings.screener import ScreenedCandidate
        from datetime import timedelta
        logger.info("\nSkipping IBKR screening, using mock candidate data")
        for e in events:
            # Mock expiry: next Friday after earnings (or +3 days if that's simpler)
            mock_expiry = e.earnings_date + timedelta(days=3)
            candidates.append(ScreenedCandidate(
                symbol=e.symbol,
                earnings_date=e.earnings_date,
                timing=e.timing,
                expiry=mock_expiry.strftime("%Y-%m-%d"),
                atm_strike=100.0,
                spot_price=100.0,
                call_bid=2.0, call_ask=2.20, call_iv=0.50,
                put_bid=2.0, put_ask=2.20, put_iv=0.50,
                straddle_mid=4.10,
                straddle_spread=0.40,
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
            # Calculate edge vs implied move (not historical average!)
            implied_move = c.implied_move_pct / 100
            edge = prediction.pred_q75 - implied_move
            status = "PASS" if edge >= args.edge_threshold else "FAIL"
            logger.info(f"{c.symbol}: q75={prediction.pred_q75:.1%}, implied={implied_move:.1%}, edge={edge:.1%} -> {status}")
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
