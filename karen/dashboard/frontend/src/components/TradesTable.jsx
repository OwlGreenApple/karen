function pnlColor(pnl) {
  if (pnl == null) return 'text-slate-500'
  if (pnl > 0) return 'text-emerald-400'
  if (pnl < 0) return 'text-red-400'
  return 'text-slate-400'
}

function fmt(ts) {
  if (!ts) return '—'
  return new Date(ts).toLocaleString('en-US', {
    timeZone: 'Asia/Jakarta',
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', hour12: false,
  })
}

function StatusBadge({ status, reason }) {
  if (status === 'open') {
    return <span className="px-1.5 py-0.5 rounded text-xs bg-indigo-900/50 text-indigo-300">OPEN</span>
  }
  const label = reason?.replace('_', ' ') ?? 'CLOSED'
  const color =
    reason === 'stop_loss' ? 'bg-red-900/50 text-red-400'
    : reason === 'take_profit' ? 'bg-emerald-900/50 text-emerald-400'
    : reason === 'time_stop' ? 'bg-amber-900/50 text-amber-400'
    : 'bg-slate-800 text-slate-400'
  return <span className={`px-1.5 py-0.5 rounded text-xs ${color}`}>{label.toUpperCase()}</span>
}

function TradeRow({ trade }) {
  const isLong = trade.side === 'long'
  return (
    <tr className="border-b border-border hover:bg-surface/50 transition-colors text-sm">
      <td className="px-3 py-2.5 text-slate-500 text-xs">{trade.id}</td>
      <td className="px-3 py-2.5 font-medium text-white">{trade.symbol}</td>
      <td className="px-3 py-2.5">
        <span className={`text-xs font-semibold ${isLong ? 'text-emerald-400' : 'text-red-400'}`}>
          {trade.side.toUpperCase()}
        </span>
      </td>
      <td className="px-3 py-2.5 text-slate-400 text-xs">{trade.strategy_mode?.replace('_', ' ')}</td>
      <td className="px-3 py-2.5 text-slate-300">{trade.entry_price?.toLocaleString('en-US', { maximumFractionDigits: 4 })}</td>
      <td className="px-3 py-2.5 text-slate-300">{trade.exit_price?.toLocaleString('en-US', { maximumFractionDigits: 4 }) ?? '—'}</td>
      <td className="px-3 py-2.5 text-slate-400">{trade.quantity?.toFixed(4)}</td>
      <td className={`px-3 py-2.5 font-semibold ${pnlColor(trade.realized_pnl)}`}>
        {trade.realized_pnl != null
          ? `${trade.realized_pnl >= 0 ? '+' : ''}$${trade.realized_pnl.toFixed(2)}`
          : '—'}
      </td>
      <td className="px-3 py-2.5">
        <StatusBadge status={trade.status} reason={trade.close_reason} />
      </td>
      <td className="px-3 py-2.5 text-slate-500 text-xs">{fmt(trade.opened_at)}</td>
    </tr>
  )
}

export default function TradesTable({ trades, total, page, pageSize, filter, onPageChange, onFilterChange }) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize))

  return (
    <div className="bg-panel rounded-xl border border-border">
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 border-b border-border">
        <h2 className="text-sm font-semibold text-slate-200">
          Trade History
          <span className="ml-2 text-xs text-slate-500 font-normal">({total})</span>
        </h2>

        <div className="flex gap-1">
          {[['', 'All'], ['open', 'Open'], ['closed', 'Closed']].map(([val, label]) => (
            <button
              key={val}
              onClick={() => onFilterChange(val)}
              className={`px-3 py-1 text-xs rounded ${
                filter === val
                  ? 'bg-indigo-600 text-white'
                  : 'bg-border text-slate-400 hover:text-white'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="text-xs text-slate-500 border-b border-border">
              {['ID', 'Symbol', 'Side', 'Mode', 'Entry', 'Exit', 'Qty', 'PnL', 'Status', 'Opened'].map((h) => (
                <th key={h} className="px-3 py-2 text-left font-normal">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {trades.length === 0 ? (
              <tr>
                <td colSpan={10} className="px-4 py-6 text-center text-slate-500 text-sm">No trades found</td>
              </tr>
            ) : (
              trades.map((t) => <TradeRow key={t.id} trade={t} />)
            )}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between px-4 py-3 border-t border-border text-sm">
          <span className="text-slate-500 text-xs">
            Page {page + 1} of {totalPages}
          </span>
          <div className="flex gap-2">
            <button
              disabled={page === 0}
              onClick={() => onPageChange(page - 1)}
              className="px-3 py-1 rounded text-xs bg-border text-slate-300 disabled:opacity-40 hover:bg-slate-700"
            >
              Prev
            </button>
            <button
              disabled={page >= totalPages - 1}
              onClick={() => onPageChange(page + 1)}
              className="px-3 py-1 rounded text-xs bg-border text-slate-300 disabled:opacity-40 hover:bg-slate-700"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
