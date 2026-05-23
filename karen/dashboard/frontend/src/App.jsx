import { useState, useEffect, useRef } from 'react'
import StatusBar from './components/StatusBar.jsx'
import EquityCard from './components/EquityCard.jsx'
import PositionsPanel from './components/PositionsPanel.jsx'
import TradesTable from './components/TradesTable.jsx'

const API = ''  // same origin via vite proxy in dev, empty in prod

export default function App() {
  const [status, setStatus] = useState(null)
  const [positions, setPositions] = useState([])
  const [trades, setTrades] = useState([])
  const [tradesTotal, setTradesTotal] = useState(0)
  const [tradesPage, setTradesPage] = useState(0)
  const [tradesFilter, setTradesFilter] = useState('')
  const [equityData, setEquityData] = useState([])
  const [equityInterval, setEquityInterval] = useState('hourly')
  const [wsConnected, setWsConnected] = useState(false)
  const wsRef = useRef(null)

  const PAGE_SIZE = 20

  async function fetchStatus() {
    try {
      const r = await fetch(`${API}/api/status`)
      if (r.ok) setStatus(await r.json())
    } catch {}
  }

  async function fetchPositions() {
    try {
      const r = await fetch(`${API}/api/positions`)
      if (r.ok) setPositions(await r.json())
    } catch {}
  }

  async function fetchTrades(page = 0, filter = '') {
    try {
      const params = new URLSearchParams({ offset: page * PAGE_SIZE, limit: PAGE_SIZE })
      if (filter) params.set('status', filter)
      const r = await fetch(`${API}/api/trades?${params}`)
      if (r.ok) {
        const data = await r.json()
        setTrades(data.trades)
        setTradesTotal(data.total)
      }
    } catch {}
  }

  async function fetchEquity(interval = 'hourly') {
    try {
      const r = await fetch(`${API}/api/equity?interval=${interval}&limit=168`)
      if (r.ok) setEquityData(await r.json())
    } catch {}
  }

  // Initial load
  useEffect(() => {
    fetchStatus()
    fetchPositions()
    fetchTrades(0, '')
    fetchEquity('hourly')
  }, [])

  // Refresh trades when page/filter changes
  useEffect(() => {
    fetchTrades(tradesPage, tradesFilter)
  }, [tradesPage, tradesFilter])

  // Refresh equity when interval changes
  useEffect(() => {
    fetchEquity(equityInterval)
  }, [equityInterval])

  // WebSocket
  useEffect(() => {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const host = window.location.host
    const ws = new WebSocket(`${proto}://${host}/ws`)
    wsRef.current = ws

    ws.onopen = () => setWsConnected(true)
    ws.onclose = () => setWsConnected(false)
    ws.onerror = () => setWsConnected(false)

    ws.onmessage = (e) => {
      try {
        const { event, data } = JSON.parse(e.data)
        if (event === 'status_update') {
          setStatus((prev) => ({ ...prev, ...data }))
        } else if (event === 'position_update') {
          setPositions(data)
        } else if (event === 'trade_opened' || event === 'trade_closed') {
          // Refresh trades and status on any trade event
          fetchTrades(tradesPage, tradesFilter)
          fetchStatus()
        } else if (event === 'equity_snapshot') {
          setEquityData((prev) => [...prev.slice(-167), data])
        }
      } catch {}
    }

    // Heartbeat ping every 30s to keep connection alive
    const ping = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send('ping')
    }, 30_000)

    return () => {
      clearInterval(ping)
      ws.close()
    }
  }, [])

  return (
    <div className="min-h-screen bg-surface">
      <StatusBar status={status} wsConnected={wsConnected} />

      <main className="max-w-7xl mx-auto px-4 py-6 space-y-6">
        <EquityCard
          data={equityData}
          interval={equityInterval}
          onIntervalChange={setEquityInterval}
        />

        <PositionsPanel positions={positions} />

        <TradesTable
          trades={trades}
          total={tradesTotal}
          page={tradesPage}
          pageSize={PAGE_SIZE}
          filter={tradesFilter}
          onPageChange={setTradesPage}
          onFilterChange={(f) => { setTradesFilter(f); setTradesPage(0) }}
        />
      </main>
    </div>
  )
}
