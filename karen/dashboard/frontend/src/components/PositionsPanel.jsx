function pnlColor(pnl) {
  if (pnl > 0) return 'text-emerald-400'
  if (pnl < 0) return 'text-red-400'
  return 'text-slate-400'
}

function PositionRow({ pos }) {
  const isLong = pos.side === 'long'
  const notional = pos.quantity * pos.mark_price
  const pnlPct = notional ? (pos.unrealized_pnl / (notional / pos.leverage)) * 100 : 0

  return (
    <div className="flex flex-wrap items-center gap-4 px-4 py-3 border-b border-border last:border-0 hover:bg-surface/50 transition-colors">
      <div className="w-28">
        <span className="font-semibold text-white text-sm">{pos.symbol}</span>
        <span className={`ml-2 text-xs font-medium px-1.5 py-0.5 rounded ${isLong ? 'bg-emerald-900/50 text-emerald-400' : 'bg-red-900/50 text-red-400'}`}>
          {pos.side.toUpperCase()}
        </span>
      </div>

      <div className="text-sm text-slate-400">
        Entry: <span className="text-slate-200">{pos.entry_price?.toLocaleString('en-US', { maximumFractionDigits: 2 })}</span>
      </div>

      <div className="text-sm text-slate-400">
        Mark: <span className="text-slate-200">{pos.mark_price?.toLocaleString('en-US', { maximumFractionDigits: 2 })}</span>
      </div>

      <div className="text-sm text-slate-400">
        Size: <span className="text-slate-200">{pos.quantity?.toFixed(4)}</span>
        <span className="text-slate-500 ml-1">({pos.leverage}x)</span>
      </div>

      <div className={`ml-auto text-sm font-semibold ${pnlColor(pos.unrealized_pnl)}`}>
        {pos.unrealized_pnl >= 0 ? '+' : ''}${pos.unrealized_pnl?.toFixed(2)}
        <span className="ml-1 text-xs font-normal">
          ({pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(2)}%)
        </span>
      </div>
    </div>
  )
}

export default function PositionsPanel({ positions }) {
  const totalPnl = positions.reduce((s, p) => s + (p.unrealized_pnl ?? 0), 0)

  return (
    <div className="bg-panel rounded-xl border border-border">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <h2 className="text-sm font-semibold text-slate-200">
          Open Positions
          <span className="ml-2 text-xs text-slate-500 font-normal">({positions.length})</span>
        </h2>
        {positions.length > 0 && (
          <span className={`text-sm font-semibold ${pnlColor(totalPnl)}`}>
            Total uPnL: {totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}
          </span>
        )}
      </div>

      {positions.length === 0 ? (
        <div className="px-4 py-6 text-center text-slate-500 text-sm">No open positions</div>
      ) : (
        positions.map((p) => <PositionRow key={p.symbol} pos={p} />)
      )}
    </div>
  )
}
