/**
 * Travel Agentic AI System — Frontend Application
 */

const API_BASE = 'http://localhost:8000';

// ==================== STATE ====================
const state = {
  currentTab: 'summary',
  travelPlan: null,
  chatOpen: false,
  chatMessages: [],
  chatSessionId: null,
  loading: false,
  selectedInterests: [],
};

const INTERESTS = [
  { id: 'culture', label: '🏛️ Culture', icon: '🏛️' },
  { id: 'food', label: '🍜 Food', icon: '🍜' },
  { id: 'adventure', label: '🏔️ Adventure', icon: '🏔️' },
  { id: 'nature', label: '🌿 Nature', icon: '🌿' },
  { id: 'nightlife', label: '🌃 Nightlife', icon: '🌃' },
  { id: 'shopping', label: '🛍️ Shopping', icon: '🛍️' },
  { id: 'history', label: '📜 History', icon: '📜' },
  { id: 'art', label: '🎨 Art', icon: '🎨' },
  { id: 'sports', label: '⚽ Sports', icon: '⚽' },
  { id: 'wellness', label: '🧘 Wellness', icon: '🧘' },
  { id: 'family', label: '👨‍👩‍👧 Family', icon: '👨‍👩‍👧' },
  { id: 'photography', label: '📸 Photography', icon: '📸' },
];

const TABS = [
  { id: 'summary', label: '📊 Summary', icon: '📊' },
  { id: 'flights', label: '✈️ Flights', icon: '✈️' },
  { id: 'hotels', label: '🏨 Hotels', icon: '🏨' },
  { id: 'places', label: '📍 Places', icon: '📍' },
  { id: 'weather', label: '🌦️ Weather', icon: '🌦️' },
  { id: 'events', label: '🎉 Events', icon: '🎉' },
  { id: 'itinerary', label: '📋 Itinerary', icon: '📋' },
  { id: 'budget', label: '💰 Budget', icon: '💰' },
];

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
  renderInterests();
  setupEventListeners();
  checkHealth();
});

function setupEventListeners() {
  document.getElementById('travelForm').addEventListener('submit', handleFormSubmit);
  document.getElementById('chatToggle').addEventListener('click', toggleChat);
  document.getElementById('chatSend').addEventListener('click', sendChatMessage);
  document.getElementById('chatInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendChatMessage();
  });
}

async function checkHealth() {
  try {
    const resp = await fetch(`${API_BASE}/api/health`);
    if (resp.ok) {
      document.getElementById('statusText').textContent = 'System Online';
      document.getElementById('statusDot').style.background = '#10b981';
    }
  } catch {
    document.getElementById('statusText').textContent = 'Connecting...';
    document.getElementById('statusDot').style.background = '#f59e0b';
  }
}

// ==================== INTERESTS ====================
function renderInterests() {
  const container = document.getElementById('interestsContainer');
  container.innerHTML = INTERESTS.map(i =>
    `<button type="button" class="interest-tag" data-interest="${i.id}" onclick="toggleInterest('${i.id}')">${i.label}</button>`
  ).join('');
}

function toggleInterest(id) {
  const idx = state.selectedInterests.indexOf(id);
  if (idx > -1) {
    state.selectedInterests.splice(idx, 1);
  } else {
    state.selectedInterests.push(id);
  }
  document.querySelectorAll('.interest-tag').forEach(tag => {
    tag.classList.toggle('active', state.selectedInterests.includes(tag.dataset.interest));
  });
}

// ==================== FORM SUBMIT ====================
async function handleFormSubmit(e) {
  e.preventDefault();
  if (state.loading) return;

  const formData = {
    source: document.getElementById('source').value.trim(),
    destination: document.getElementById('destination').value.trim(),
    departure_date: document.getElementById('departureDate').value,
    return_date: document.getElementById('returnDate').value,
    travelers: parseInt(document.getElementById('travelers').value) || 1,
    interests: state.selectedInterests.length > 0 ? state.selectedInterests : ['culture', 'food'],
    budget_level: document.getElementById('budgetLevel').value,
    budget_amount: parseFloat(document.getElementById('budgetAmount').value) || null,
    include_events: true,
    natural_language_input: document.getElementById('specialReqs').value || null,
  };

  if (!formData.source || !formData.destination || !formData.departure_date || !formData.return_date) {
    showNotification('Please fill in all required fields', 'error');
    return;
  }

  state.loading = true;
  showLoading();

  try {
    const resp = await fetch(`${API_BASE}/api/plan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData),
    });

    if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
    const data = await resp.json();
    state.travelPlan = data;
    hideLoading();
    renderResults(data);
    showNotification('✨ Travel plan generated successfully!', 'success');
  } catch (err) {
    hideLoading();
    console.error('Planning error:', err);
    showNotification('Failed to generate plan. Make sure the backend is running.', 'error');
  } finally {
    state.loading = false;
  }
}

// ==================== LOADING ====================
function showLoading() {
  const overlay = document.getElementById('loadingOverlay');
  overlay.classList.add('active');
  animateAgents();
}

function hideLoading() {
  document.getElementById('loadingOverlay').classList.remove('active');
}

function animateAgents() {
  const agents = ['✈️ Flight', '🏨 Hotel', '📍 Place', '🌦️ Weather', '🎉 Event', '💰 Budget', '📏 Distance', '📋 Itinerary'];
  const container = document.getElementById('loadingAgents');
  container.innerHTML = agents.map(a => `<div class="agent-chip">${a}</div>`).join('');

  let idx = 0;
  const chips = container.querySelectorAll('.agent-chip');
  const interval = setInterval(() => {
    if (idx > 0) chips[idx - 1].classList.replace('active', 'done');
    if (idx < chips.length) {
      chips[idx].classList.add('active');
      document.getElementById('loadingStatus').textContent = `${agents[idx]} agent processing...`;
      idx++;
    } else {
      clearInterval(interval);
      document.getElementById('loadingStatus').textContent = 'Generating your travel plan...';
    }
  }, 800);
}

// ==================== RENDER RESULTS ====================
function renderResults(data) {
  const container = document.getElementById('resultsContainer');
  container.classList.add('active');

  // Header
  document.getElementById('resultsTitle').textContent = `${data.source} → ${data.destination}`;
  renderTabs();
  switchTab('summary');

  // Scroll to results
  container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderTabs() {
  const nav = document.getElementById('tabNavigation');
  nav.innerHTML = TABS.map(t =>
    `<button class="tab-btn ${t.id === state.currentTab ? 'active' : ''}" onclick="switchTab('${t.id}')" id="tab-${t.id}">${t.label}</button>`
  ).join('');
}

function switchTab(tabId) {
  state.currentTab = tabId;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById(`tab-${tabId}`);
  if (btn) btn.classList.add('active');

  const panel = document.getElementById('tabContent');
  const data = state.travelPlan;
  if (!data) return;

  const renderers = {
    summary: () => renderSummary(data),
    flights: () => renderFlights(data.flights || []),
    hotels: () => renderHotels(data.hotels || []),
    places: () => renderPlaces(data.places || []),
    weather: () => renderWeather(data.weather || []),
    events: () => renderEvents(data.events || []),
    itinerary: () => renderItinerary(data.itinerary || []),
    budget: () => renderBudget(data.budget || {}),
  };

  panel.innerHTML = (renderers[tabId] || renderers.summary)();
}

// ==================== TAB RENDERERS ====================

function renderSummary(data) {
  const insights = (data.ai_insights || []).map(i =>
    `<div class="insight-card"><div class="insight-icon">💡</div>${i}</div>`
  ).join('');
  const personNotes = (data.personalization_notes || []).map(n =>
    `<div class="insight-card"><div class="insight-icon">🎯</div>${n}</div>`
  ).join('');

  return `
    <div class="summary-banner">
      <h3>🌍 Your AI-Crafted Travel Plan</h3>
      <div class="summary-text">${data.travel_summary || 'Your personalized travel plan is ready!'}</div>
      <div class="summary-meta">
        <div class="meta-item"><span class="icon">📅</span><strong>${data.departure_date}</strong> to <strong>${data.return_date}</strong></div>
        <div class="meta-item"><span class="icon">👥</span><strong>${data.travelers}</strong> traveler(s)</div>
        <div class="meta-item"><span class="icon">⏱️</span>Generated in <strong>${data.processing_time || 0}s</strong></div>
        <div class="meta-item"><span class="icon">🆔</span>${data.request_id || ''}</div>
      </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:12px;margin-bottom:32px;">
      <div class="glass-card" style="text-align:center;padding:20px;">
        <div style="font-size:1.8rem;">✈️</div>
        <div style="font-size:1.5rem;font-weight:700;font-family:var(--font-heading);">${(data.flights || []).length}</div>
        <div style="font-size:0.8rem;color:var(--text-muted);">Flights</div>
      </div>
      <div class="glass-card" style="text-align:center;padding:20px;">
        <div style="font-size:1.8rem;">🏨</div>
        <div style="font-size:1.5rem;font-weight:700;font-family:var(--font-heading);">${(data.hotels || []).length}</div>
        <div style="font-size:0.8rem;color:var(--text-muted);">Hotels</div>
      </div>
      <div class="glass-card" style="text-align:center;padding:20px;">
        <div style="font-size:1.8rem;">📍</div>
        <div style="font-size:1.5rem;font-weight:700;font-family:var(--font-heading);">${(data.places || []).length}</div>
        <div style="font-size:0.8rem;color:var(--text-muted);">Places</div>
      </div>
      <div class="glass-card" style="text-align:center;padding:20px;">
        <div style="font-size:1.8rem;">🎉</div>
        <div style="font-size:1.5rem;font-weight:700;font-family:var(--font-heading);">${(data.events || []).length}</div>
        <div style="font-size:0.8rem;color:var(--text-muted);">Events</div>
      </div>
      <div class="glass-card" style="text-align:center;padding:20px;">
        <div style="font-size:1.8rem;">📋</div>
        <div style="font-size:1.5rem;font-weight:700;font-family:var(--font-heading);">${(data.itinerary || []).length}</div>
        <div style="font-size:0.8rem;color:var(--text-muted);">Days Planned</div>
      </div>
    </div>
    ${insights || personNotes ? `
      <h3 style="font-family:var(--font-heading);margin-bottom:16px;">🧠 AI Insights</h3>
      <div class="insights-grid">${insights}${personNotes}</div>
    ` : ''}
  `;
}

function renderFlights(flights) {
  if (!flights.length) return '<div class="glass-card"><p style="color:var(--text-muted);">No flights found.</p></div>';
  return `<div class="card-grid cols-2">${flights.map(f => {
    const depTime = f.departure_time ? f.departure_time.split('T')[1]?.substring(0, 5) || '' : '';
    const arrTime = f.arrival_time ? f.arrival_time.split('T')[1]?.substring(0, 5) || '' : '';
    return `
    <div class="flight-card">
      <div class="score-badge">${Math.round((f.score || 0) * 100)}% match</div>
      <div class="flight-header">
        <div class="airline-icon">✈️</div>
        <div class="airline-info">
          <h4>${f.airline || 'Airline'}</h4>
          <span>${f.flight_number || ''}</span>
        </div>
      </div>
      <div class="flight-route">
        <div class="airport">
          <div class="code">${f.departure_airport || '---'}</div>
          <div class="time">${depTime}</div>
        </div>
        <div class="route-line">
          <div class="duration">${f.duration || ''}</div>
          <div class="line"></div>
          <div class="stops">${f.stops === 0 ? 'Non-stop' : f.stops + ' stop(s)'}</div>
        </div>
        <div class="airport">
          <div class="code">${f.arrival_airport || '---'}</div>
          <div class="time">${arrTime}</div>
        </div>
      </div>
      <div class="flight-footer">
        <div class="flight-price">$${(f.price || 0).toLocaleString()}</div>
        <div class="flight-class">${f.cabin_class || 'economy'}</div>
      </div>
    </div>`;
  }).join('')}</div>`;
}

function renderHotels(hotels) {
  if (!hotels.length) return '<div class="glass-card"><p style="color:var(--text-muted);">No hotels found.</p></div>';
  return `<div class="card-grid cols-2">${hotels.map(h => `
    <div class="hotel-card">
      <div class="hotel-header">
        <div>
          <div class="hotel-name">${h.name || 'Hotel'}</div>
          <div class="hotel-stars">${'⭐'.repeat(h.stars || 3)}</div>
        </div>
        <div class="hotel-rating">${(h.rating || 0).toFixed(1)}</div>
      </div>
      <div class="hotel-address">📍 ${h.address || ''} ${h.distance_to_center ? `• ${h.distance_to_center} km from center` : ''}</div>
      <div class="amenities">${(h.amenities || []).slice(0, 6).map(a => `<span class="amenity-tag">${a}</span>`).join('')}</div>
      <div class="hotel-footer">
        <div class="hotel-price">$${(h.price_per_night || 0).toLocaleString()} <span>/night</span></div>
        <div class="hotel-distance">Score: ${Math.round((h.score || 0) * 100)}%</div>
      </div>
    </div>
  `).join('')}</div>`;
}

function renderPlaces(places) {
  if (!places.length) return '<div class="glass-card"><p style="color:var(--text-muted);">No places found.</p></div>';
  return `<div class="card-grid cols-3">${places.map(p => `
    <div class="place-card">
      <div class="place-header">
        <div class="place-name">${p.name || 'Place'}</div>
        <span class="place-category">${p.category || ''}</span>
      </div>
      <div class="place-rating">${'⭐'.repeat(Math.round(p.rating || 0))} ${(p.rating || 0).toFixed(1)}</div>
      <div class="place-desc">${p.description || 'A must-visit attraction.'}</div>
      ${p.tips && p.tips.length ? `<div class="place-tips"><p>💡 ${p.tips[0]}</p></div>` : ''}
    </div>
  `).join('')}</div>`;
}

function renderWeather(weather) {
  if (!weather.length) return '<div class="glass-card"><p style="color:var(--text-muted);">No weather data.</p></div>';
  const iconMap = { 'Clear': '☀️', 'Clouds': '⛅', 'Rain': '🌧️', 'Drizzle': '🌦️', 'Thunderstorm': '⛈️', 'Snow': '❄️', 'Mist': '🌫️', 'Fog': '🌫️' };
  return `<div class="weather-grid">${weather.map(w => `
    <div class="weather-card">
      <div class="weather-date">${formatDate(w.date)}</div>
      <div class="weather-icon">${iconMap[w.condition] || '🌤️'}</div>
      <div class="weather-temp">${Math.round(w.temperature_high)}° <span class="low">/ ${Math.round(w.temperature_low)}°</span></div>
      <div class="weather-desc">${w.description || w.condition}</div>
      <div class="weather-details">
        <span>💧 ${w.humidity}%</span>
        <span>💨 ${w.wind_speed} m/s</span>
      </div>
      <div class="weather-rec">${w.recommendation || ''}</div>
    </div>
  `).join('')}</div>`;
}

function renderEvents(events) {
  if (!events.length) return '<div class="glass-card"><p style="color:var(--text-muted);">No events found.</p></div>';
  return `<div class="card-grid cols-2">${events.map(e => `
    <div class="event-card">
      <div class="event-type">${e.event_type || 'Event'}</div>
      <div class="event-name">${e.name || 'Event'}</div>
      <div class="event-venue">📍 ${e.venue || ''}</div>
      <div class="event-datetime">
        <span>📅 ${formatDate(e.date)}</span>
        <span>🕐 ${e.time || ''}</span>
      </div>
      <div style="font-size:0.85rem;color:var(--text-secondary);margin:8px 0;">${e.description || ''}</div>
      <div class="event-price">🎟️ ${e.price_range || 'Free'}</div>
      ${e.url ? `<a href="${e.url}" target="_blank" style="color:var(--accent-tertiary);font-size:0.8rem;text-decoration:none;display:inline-block;margin-top:8px;">View Details →</a>` : ''}
    </div>
  `).join('')}</div>`;
}

function renderItinerary(itinerary) {
  if (!itinerary.length) return '<div class="glass-card"><p style="color:var(--text-muted);">No itinerary generated.</p></div>';
  return `<div class="itinerary-timeline">${itinerary.map(day => {
    const activities = (day.activities || []).map(a => `
      <div class="activity-item">
        <div class="activity-time">${a.time || ''}</div>
        <div class="activity-details">
          <h5>${a.activity || a.name || ''}</h5>
          <p>${a.description || ''}</p>
          <div class="activity-meta">
            ${a.location ? `<span>📍 ${a.location}</span>` : ''}
            ${a.duration ? `<span>⏱️ ${a.duration}</span>` : ''}
            ${a.cost_estimate ? `<span>💰 $${a.cost_estimate}</span>` : ''}
          </div>
        </div>
      </div>
    `).join('');

    const meals = (day.meals || []).map(m => `
      <div class="activity-item">
        <div class="activity-time">${m.time || ''}</div>
        <div class="activity-details">
          <h5>${m.meal_type || ''}: ${m.suggestion || ''}</h5>
          <div class="activity-meta">
            <span>🍽️ ${m.cuisine || ''}</span>
            ${m.estimated_cost ? `<span>💰 $${m.estimated_cost}</span>` : ''}
          </div>
        </div>
      </div>
    `).join('');

    return `
    <div class="itinerary-day">
      <div class="day-header">
        <span class="day-number">Day ${day.day_number || ''}</span>
        <span class="day-date">${formatDate(day.date)}</span>
        <span class="day-theme">${day.theme || ''}</span>
      </div>
      <div class="day-content">
        ${day.weather_summary ? `<div class="day-weather">${day.weather_summary}</div>` : ''}
        <div class="activity-list">${activities}${meals}</div>
        ${(day.travel_tips || []).length ? `
          <div style="margin-top:16px;padding:12px;background:rgba(99,102,241,0.05);border-radius:8px;">
            <strong style="font-size:0.85rem;">💡 Tips:</strong>
            <ul style="margin:8px 0 0 16px;font-size:0.8rem;color:var(--text-secondary);">
              ${day.travel_tips.map(t => `<li>${t}</li>`).join('')}
            </ul>
          </div>
        ` : ''}
      </div>
    </div>`;
  }).join('')}</div>`;
}

function renderBudget(budget) {
  if (!budget || !budget.total) {
    return '<div class="glass-card"><p style="color:var(--text-muted);">No budget data available.</p></div>';
  }
  const categories = [
    { key: 'flights', label: '✈️ Flights', color: '#6366f1' },
    { key: 'accommodation', label: '🏨 Accommodation', color: '#8b5cf6' },
    { key: 'food', label: '🍽️ Food & Dining', color: '#f59e0b' },
    { key: 'activities', label: '🎯 Activities', color: '#10b981' },
    { key: 'transportation', label: '🚗 Transportation', color: '#3b82f6' },
    { key: 'events', label: '🎉 Events', color: '#ec4899' },
    { key: 'miscellaneous', label: '📦 Miscellaneous', color: '#64748b' },
  ];
  const maxVal = Math.max(...categories.map(c => budget[c.key] || 0), 1);
  const bars = categories.map(c => {
    const val = budget[c.key] || 0;
    const pct = (val / maxVal * 100).toFixed(0);
    return `
    <div class="budget-bar">
      <div class="label">${c.label}</div>
      <div class="bar-container"><div class="bar-fill" style="width:${pct}%;background:${c.color};"></div></div>
      <div class="amount">$${val.toLocaleString()}</div>
    </div>`;
  }).join('');

  const tips = (budget.savings_tips || []).map(t =>
    `<div class="tip-item"><span class="tip-icon">💡</span><span>${t}</span></div>`
  ).join('');

  return `
    <div class="budget-overview">
      <div class="budget-chart">
        <h3>💰 Budget Breakdown</h3>
        ${bars}
        <div class="budget-total">
          <div class="total-label">Total Estimated Cost</div>
          <div class="total-amount">$${(budget.total || 0).toLocaleString()}</div>
        </div>
      </div>
      <div class="budget-tips">
        <h3>💡 Money-Saving Tips</h3>
        ${tips || '<p style="color:var(--text-muted);">No tips available.</p>'}
      </div>
    </div>
  `;
}

// ==================== CHAT ====================
function toggleChat() {
  state.chatOpen = !state.chatOpen;
  const panel = document.getElementById('chatPanel');
  panel.classList.toggle('active', state.chatOpen);
  if (state.chatOpen && !state.chatMessages.length) {
    addChatMessage('assistant', '👋 Hi! I\'m your AI travel assistant. Ask me anything about your trip!');
  }
}

async function sendChatMessage() {
  const input = document.getElementById('chatInput');
  const msg = input.value.trim();
  if (!msg) return;

  input.value = '';
  addChatMessage('user', msg);

  try {
    const resp = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: msg,
        session_id: state.chatSessionId,
        travel_context: state.travelPlan ? {
          destination: state.travelPlan.destination,
          dates: `${state.travelPlan.departure_date} to ${state.travelPlan.return_date}`,
        } : null,
      }),
    });
    const data = await resp.json();
    state.chatSessionId = data.session_id;
    addChatMessage('assistant', data.response);
  } catch {
    addChatMessage('assistant', 'Sorry, I\'m having trouble connecting. Please check the backend server.');
  }
}

function addChatMessage(role, content) {
  state.chatMessages.push({ role, content });
  const container = document.getElementById('chatMessages');
  const msgEl = document.createElement('div');
  msgEl.className = `chat-message ${role}`;
  msgEl.textContent = content;
  container.appendChild(msgEl);
  container.scrollTop = container.scrollHeight;
}

// ==================== UTILITIES ====================
function formatDate(dateStr) {
  if (!dateStr) return '';
  try {
    const d = new Date(dateStr + 'T00:00:00');
    return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
  } catch {
    return dateStr;
  }
}

function showNotification(message, type = 'info') {
  const existing = document.querySelector('.notification');
  if (existing) existing.remove();

  const colors = { success: '#10b981', error: '#ef4444', info: '#3b82f6', warning: '#f59e0b' };
  const el = document.createElement('div');
  el.className = 'notification';
  el.textContent = message;
  Object.assign(el.style, {
    position: 'fixed', top: '90px', right: '24px', padding: '14px 24px',
    background: `${colors[type]}20`, border: `1px solid ${colors[type]}40`,
    color: colors[type], borderRadius: '12px', zIndex: '300',
    fontFamily: 'var(--font-body)', fontSize: '0.9rem', fontWeight: '500',
    backdropFilter: 'blur(8px)', animation: 'fadeIn 0.3s ease',
    boxShadow: `0 4px 16px ${colors[type]}20`,
  });
  document.body.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; el.style.transition = '0.3s'; setTimeout(() => el.remove(), 300); }, 4000);
}

// Expose to global
window.toggleInterest = toggleInterest;
window.switchTab = switchTab;
