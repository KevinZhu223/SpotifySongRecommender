// Main JavaScript for Spotify Recommender

let userData = null;
let userTracks = [];

// Initialize the application
function initializeApp() {
    console.log('Initializing Spotify Recommender...');
    
    // Set up event listeners
    setupEventListeners();
    
    // Check if data is already available
    checkDataStatus();
}

// Set up event listeners
function setupEventListeners() {
    // Data collection
    $('#collectDataBtn').click(collectUserData);
    
    // Profile viewing
    $('#viewProfileBtn').click(showUserProfile);
    
    // Recommendations
    $('#getRecommendationsBtn').click(getRecommendations);
    $('#recommendationType').change(handleRecommendationTypeChange);
    
    // Search
    $('#searchBtn').click(searchTracks);
    $('#searchQuery').keypress(function(e) {
        if (e.which === 13) { // Enter key
            searchTracks();
        }
    });
}

// Check if user data is already available
function checkDataStatus() {
    $.get('/health')
        .done(function(response) {
            console.log('Health check:', response);
            if (response.data_loaded) {
                enableRecommendationFeatures();
            }
        })
        .fail(function() {
            console.log('Health check failed — server may not be running');
        });
}

// Collect user data
function collectUserData() {
    const btn = $('#collectDataBtn');
    const originalText = btn.html();
    
    // Show loading state
    btn.prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Collecting Data...');
    showLoadingModal('Collecting your Spotify data...', 'This may take a few moments');
    
    $.post('/collect_data')
        .done(function(response) {
            hideLoadingModal();
            
            if (response.success) {
                userData = response.profile;
                updateProfileDisplay();
                enableRecommendationFeatures();
                loadUserTracks();
                
                showAlert('Data collected successfully!', 'success');
            } else {
                showAlert(response.message, 'danger');
            }
        })
        .fail(function(xhr) {
            hideLoadingModal();
            const errorMessage = xhr.responseJSON ? xhr.responseJSON.message : 'Error collecting data';
            showAlert(errorMessage, 'danger');
        })
        .always(function() {
            btn.prop('disabled', false).html(originalText);
        });
}

// Update profile display
function updateProfileDisplay() {
    if (!userData) return;
    
    $('#totalTracks').text(userData.total_tracks || 0);
    $('#totalArtists').text(userData.unique_artists || 0);
    $('#totalAlbums').text(userData.unique_albums || 0);
    
    $('#profileInfo').removeClass('d-none');
    $('#viewProfileBtn').prop('disabled', false);
}

// Enable recommendation features
function enableRecommendationFeatures() {
    $('#getRecommendationsBtn').prop('disabled', false);
    loadUserTracks();
}

// Load user tracks for selection
function loadUserTracks() {
    $.get('/user_tracks')
        .done(function(response) {
            if (response.success) {
                userTracks = response.tracks;
                populateTrackSelect();
            }
        })
        .fail(function() {
            console.log('Error loading user tracks');
        });
}

// Populate track selection dropdown
function populateTrackSelect() {
    const select = $('#trackSelect');
    select.empty().append('<option value="">Select a track...</option>');
    
    userTracks.forEach(track => {
        const option = $(`<option value="${track.track_id}">${track.track_name} - ${track.artist_name}</option>`);
        select.append(option);
    });
}

// Handle recommendation type change
function handleRecommendationTypeChange() {
    const type = $('#recommendationType').val();
    const trackSelect = $('#trackSelect');
    
    if (type === 'content_based' || type === 'hybrid') {
        trackSelect.prop('disabled', false);
    } else {
        trackSelect.prop('disabled', true);
    }
}

// Get recommendations
function getRecommendations() {
    const type = $('#recommendationType').val();
    const numRecs = $('#numRecommendations').val();
    const trackId = $('#trackSelect').val();
    
    // Validate inputs
    if ((type === 'content_based' || type === 'hybrid') && !trackId) {
        showAlert('Please select a track for content-based or hybrid recommendations', 'warning');
        return;
    }
    
    const btn = $('#getRecommendationsBtn');
    const originalText = btn.html();
    
    // Show loading state
    btn.prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Getting Recommendations...');
    
    // Build URL
    let url = `/recommendations?type=${type}&n=${numRecs}`;
    if (trackId) {
        url += `&track_id=${trackId}`;
    }
    
    $.get(url)
        .done(function(response) {
            if (response.success) {
                displayRecommendations(response.recommendations, response.type);
                showAlert(`Found ${response.recommendations.length} recommendations`, 'success');
            } else {
                showAlert(response.message, 'danger');
            }
        })
        .fail(function(xhr) {
            const errorMessage = xhr.responseJSON ? xhr.responseJSON.message : 'Error getting recommendations';
            showAlert(errorMessage, 'danger');
        })
        .always(function() {
            btn.prop('disabled', false).html(originalText);
        });
}

// Display recommendations
function displayRecommendations(recommendations, type) {
    const container = $('#recommendationsList');
    const section = $('#recommendationsSection');
    
    container.empty();
    
    if (recommendations.length === 0) {
        container.html('<div class="col-12"><p class="text-muted text-center">No recommendations found.</p></div>');
    } else {
        recommendations.forEach((rec, index) => {
            const card = createRecommendationCard(rec, index + 1);
            container.append(card);
        });
    }
    
    section.removeClass('d-none').addClass('fade-in');
    
    // Scroll to recommendations
    $('html, body').animate({
        scrollTop: section.offset().top - 100
    }, 500);
}

// Create recommendation card
function createRecommendationCard(recommendation, index) {
    const similarityPercent = Math.round(recommendation.similarity_score * 100);
    
    return $(`
        <div class="col-md-6 col-lg-4 mb-3">
            <div class="track-card">
                <div class="track-info">
                    <h6>${recommendation.track_name}</h6>
                    <p><strong>Artist:</strong> ${recommendation.artist_name}</p>
                    <p><strong>Album:</strong> ${recommendation.album_name}</p>
                </div>
                <div class="track-meta">
                    <span class="similarity-score">${similarityPercent}% match</span>
                    <span class="recommendation-type">${recommendation.recommendation_type}</span>
                </div>
            </div>
        </div>
    `);
}

// Show user profile
function showUserProfile() {
    if (!userData) {
        showAlert('No user data available', 'warning');
        return;
    }
    
    const modal = $('#profileModal');
    const content = $('#profileContent');
    
    // Build profile content
    let html = '<div class="row">';
    
    // Basic stats
    html += `
        <div class="col-md-4">
            <div class="profile-stat">
                <h3>${userData.total_tracks || 0}</h3>
                <p>Total Tracks</p>
            </div>
        </div>
        <div class="col-md-4">
            <div class="profile-stat">
                <h3>${userData.unique_artists || 0}</h3>
                <p>Unique Artists</p>
            </div>
        </div>
        <div class="col-md-4">
            <div class="profile-stat">
                <h3>${userData.unique_albums || 0}</h3>
                <p>Unique Albums</p>
            </div>
        </div>
    `;
    
    // Audio features
    const audioFeatures = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'liveness', 'speechiness'];
    html += '<div class="col-12 mt-4"><h5>Audio Features</h5></div>';
    
    audioFeatures.forEach(feature => {
        const avgKey = `avg_${feature}`;
        if (userData[avgKey] !== undefined) {
            const value = Math.round(userData[avgKey] * 100);
            html += `
                <div class="col-md-6">
                    <div class="audio-feature">
                        <div class="audio-feature-label">${feature.charAt(0).toUpperCase() + feature.slice(1)}</div>
                        <div class="progress">
                            <div class="progress-bar" style="width: ${value}%"></div>
                        </div>
                        <small class="text-muted">${value}%</small>
                    </div>
                </div>
            `;
        }
    });
    
    // Top artists
    if (userData.top_artists) {
        html += '<div class="col-12 mt-4"><h5>Top Artists</h5></div>';
        Object.entries(userData.top_artists).slice(0, 10).forEach(([artist, count]) => {
            html += `
                <div class="col-md-6">
                    <div class="search-result">
                        <strong>${artist}</strong>
                        <span class="badge bg-primary ms-2">${count} tracks</span>
                    </div>
                </div>
            `;
        });
    }
    
    html += '</div>';
    content.html(html);
    modal.modal('show');
}

// Search tracks
function searchTracks() {
    const query = $('#searchQuery').val().trim();
    
    if (!query) {
        showAlert('Please enter a search query', 'warning');
        return;
    }
    
    const btn = $('#searchBtn');
    const originalText = btn.html();
    
    // Show loading state
    btn.prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i>');
    
    $.get(`/search?q=${encodeURIComponent(query)}&n=10`)
        .done(function(response) {
            if (response.success) {
                displaySearchResults(response.results, response.query);
            } else {
                showAlert(response.message, 'danger');
            }
        })
        .fail(function(xhr) {
            const errorMessage = xhr.responseJSON ? xhr.responseJSON.message : 'Error searching tracks';
            showAlert(errorMessage, 'danger');
        })
        .always(function() {
            btn.prop('disabled', false).html(originalText);
        });
}

// Display search results
function displaySearchResults(results, query) {
    const container = $('#searchResults');
    
    if (results.length === 0) {
        container.html(`<p class="text-muted">No tracks found for "${query}"</p>`);
        return;
    }
    
    let html = `<h6>Search Results for "${query}"</h6>`;
    
    results.forEach(result => {
        html += `
            <div class="search-result">
                <h6>${result.track_name}</h6>
                <p class="mb-1"><strong>Artist:</strong> ${result.artist_name}</p>
                <p class="mb-0"><strong>Album:</strong> ${result.album_name}</p>
            </div>
        `;
    });
    
    container.html(html);
}

// Show loading modal
function showLoadingModal(title, subtitle) {
    $('#loadingText').text(title);
    $('#loadingSubtext').text(subtitle);
    $('#loadingModal').modal('show');
}

// Hide loading modal
function hideLoadingModal() {
    $('#loadingModal').modal('hide');
}

// Show alert
function showAlert(message, type) {
    const alertId = 'alert-' + Date.now();
    const alert = $(`
        <div id="${alertId}" class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `);
    
    $('#alertContainer').append(alert);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        $(`#${alertId}`).alert('close');
    }, 5000);
}

// Utility functions
function formatDuration(ms) {
    const minutes = Math.floor(ms / 60000);
    const seconds = Math.floor((ms % 60000) / 1000);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}
