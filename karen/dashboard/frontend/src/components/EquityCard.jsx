import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts'

function fmt(ts) {
  const d = new Date(ts)
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false })
}

function fmtShort(ts) {
  const d = new Date(ts)
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  const { equity_usdt, unrealized_pnl } = payload[0].payload
  return (
    <div className="bg-panel border border-border rounded px-3 py-2 text-xs">
      <p className="text-slate-400">{fmt(label)}</p>
      <p className="text-white font-semibold">${equity_usdt?.toLocaleString('en-US', { maximumFractionDigits: 2 })}</p>
      {unrealized_pnl != null && (
        <p className={unrealized_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}>
          uPnL: ${unrealized_pnl >= 0 ? '+' : ''}{unrealized_pnl?.toFixed(2)}
        </p>
      )}
    </div>
  )
}

export default function EquityCard({ data, interval, onIntervalChange }) {
  const latest = data[data.length - 1]
  const first = data[0]
  const change = latest && first ? latest.equity_usdt - first.equity_usdt : 0
  const changePct = first?.equity_usdt ? (change / first.equity_usdt) * 100 : 0

  return (
    <div className="bg-panel rounded-xl border border-border p-5">
      <div className="flex items-start justify-between mb-4">
        <div>
          <p className="text-slate-400 text-sm mb-1">Portfolio Equity</p>
          <p className="text-2xl font-bold text-white">
            ${latest?.equity_usdt?.toLocaleString('en-US', { maximumFractionDigits: 2 }) ?? '—'}
          </p>
          {data.length > 1 && (
            <p className={`text-sm mt-0.5 ${change >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {change >= 0 ? '+' : ''}${change.toFixed(2)} ({changePct >= 0 ? '+' : ''}{changePct.toFixed(2)}%)
            </p>
          )}
        </div>

        <div className="flex gap-1">
          {['hourly', 'daily'].map((iv) => (
            <button
              key={iv}
              onClick={() => onIntervalChange(iv)}
              className={`px-3 py-1 text-xs rounded ${
                interval === iv
                  ? 'bg-indigo-600 text-white'
                  : 'bg-border text-slate-400 hover:text-white'
              }`}
            >
              {iv === 'hourly' ? '1H' : '1D'}
            </button>
          ))}
        </div>
      </div>

      {data.length === 0 ? (
        <div className="h-40 flex items-center justify-center text-slate-500 text-sm">
          No equity snapshots yet
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={160}>
          <AreaChart data={data} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" vertical={false} />
            <XAxis
              dataKey="timestamp"
              tickFormatter={interval === 'daily' ? fmtShort : (ts) => new Date(ts).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })}
              tick={{ fill: '#64748b', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={['auto', 'auto']}
              tick={{ fill: '#64748b', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              width={60}
              tickFormatter={(v) => `$${(v / 1000).toFixed(1)}k`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="equity_usdt"
              stroke="#6366f1"
              strokeWidth={2}
              fill="url(#equityGrad)"
              dot={false}
              activeDot={{ r: 4, fill: '#6366f1' }}
            />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
