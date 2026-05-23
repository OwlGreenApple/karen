export default function StatusBar({ status, wsConnected }) {
  if (!status) {
    return (
      <header className="bg-panel border-b border-border px-6 py-3 flex items-center gap-3">
        <span className="text-lg font-bold text-white">Karen</span>
        <span className="text-slate-500 text-sm">Loading...</span>
      </header>
    )
  }

  const tradingColor = status.trading_enabled
    ? 'text-emerald-400'
    : status.paused
    ? 'text-amber-400'
    : 'text-slate-400'

  const tradingLabel = status.trading_enabled
    ? status.paused
      ? 'PAUSED'
      : 'LIVE'
    : 'DISABLED'

  return (
    <header className="bg-panel border-b border-border px-6 py-3 flex flex-wrap items-center gap-4">
      <span className="text-lg font-bold text-white tracking-wide">Karen</span>

      <div className="flex items-center gap-1.5">
        <span className={`text-xs font-semibold uppercase tracking-wider ${tradingColor}`}>
          {tradingLabel}
        </span>
      </div>

      <div className="text-slate-400 text-sm">
        Mode: <span className="text-slate-200 font-medium">{status.strategy_mode.replace('_', ' ')}</span>
      </div>

      <div className="text-slate-400 text-sm">
        Equity: <span className="text-slate-200 font-medium">${status.equity_usdt?.toLocaleString('en-US', { maximumFractionDigits: 2 })}</span>
      </div>

      <div className="ml-auto flex items-center gap-2">
        <span className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-emerald-500' : 'bg-red-500'}`} />
        <span className="text-xs text-slate-500">{wsConnected ? 'Live' : 'Offline'}</span>
      </div>
    </header>
  )
}
