import numpy as np
import pandas as pd
import re
import urllib.parse
import tldextract
import whois
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler
import joblib
import time
from urllib.parse import urlparse
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PhishingDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def _extract_url_features(self, url):
        """Extract features from the URL itself"""
        features = {}

        # Parse URL
        parsed_url = urlparse(url)
        path = parsed_url.path
        query = parsed_url.query
        fragment = parsed_url.fragment

        # URL length features
        features['url_length'] = len(url)
        features['domain_length'] = len(parsed_url.netloc)
        features['path_length'] = len(path)
        features['query_length'] = len(query)

        # Character distribution features
        features['digit_ratio'] = sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0
        features['letter_ratio'] = sum(c.isalpha() for c in url) / len(url) if len(url) > 0 else 0
        features['non_alnum_ratio'] = sum(not c.isalnum() for c in url) / len(url) if len(url) > 0 else 0

        # Special character counts
        features['dot_count'] = url.count('.')
        features['hyphen_count'] = url.count('-')
        features['underscore_count'] = url.count('_')
        features['slash_count'] = url.count('/')
        features['question_count'] = url.count('?')
        features['equal_count'] = url.count('=')
        features['at_count'] = url.count('@')
        features['and_count'] = url.count('&')
        features['exclamation_count'] = url.count('!')
        features['space_count'] = url.count(' ')
        features['tilde_count'] = url.count('~')
        features['comma_count'] = url.count(',')
        features['plus_count'] = url.count('+')
        features['asterisk_count'] = url.count('*')
        features['hashtag_count'] = url.count('#')
        features['dollar_count'] = url.count('$')
        features['percent_count'] = url.count('%')

        # Path and query features
        features['dir_count'] = path.count('/')
        features['param_count'] = query.count('&') + 1 if query else 0

        # Presence of sensitive terms in URL (keep minimal)
        features['has_login'] = 1 if re.search(r'login|signin|log-in|sign-in', url.lower()) else 0
        features['has_secure'] = 1 if re.search(r'secure|safe|protect', url.lower()) else 0

        # Protocol features
        features['is_https'] = 1 if url.startswith('https://') else 0

        # Numerical sequences
        digit_sequences = re.findall(r'\d+', url)
        features['max_digit_sequence_length'] = max([len(seq) for seq in digit_sequences]) if digit_sequences else 0

        return features

    def _extract_domain_features(self, url):
        """Extract features related to the domain"""
        features = {}

        # Extract domain parts
        extracted = tldextract.extract(url)
        domain = extracted.domain
        suffix = extracted.suffix
        subdomain = extracted.subdomain

        # Domain features
        features['domain_label_count'] = len(domain.split('.'))
        features['has_subdomain'] = 1 if subdomain else 0
        features['subdomain_length'] = len(subdomain)
        features['subdomain_count'] = subdomain.count('.') + 1 if subdomain else 0

        # TLD features
        features['tld_length'] = len(suffix)

        # IP address features
        features['is_ip_address'] = 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', urlparse(url).netloc) else 0

        # Port features
        port_match = re.search(r':(\d+)', urlparse(url).netloc)
        features['has_port'] = 1 if port_match else 0
        features['port_number'] = int(port_match.group(1)) if port_match else -1

        # Common TLDs vs unusual ones
        common_tlds = ['com', 'org', 'net', 'edu', 'gov', 'info', 'io', 'co']
        features['is_common_tld'] = 1 if suffix in common_tlds else 0

        return features

    def _extract_content_features(self, url, timeout=3):
        """Extract features from webpage content with timeout"""
        features = {}

        try:
            # Set a timeout for requests
            response = requests.get(url, timeout=timeout,
                                     headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Basic content features
                features['title_length'] = len(soup.title.text) if soup.title else 0
                features['body_length'] = len(soup.body.text) if soup.body else 0

                # Form and input related features (highly relevant for phishing)
                features['form_count'] = len(soup.find_all('form'))
                features['input_count'] = len(soup.find_all('input'))
                features['password_input_count'] = len(soup.find_all('input', {'type': 'password'}))

                # Iframe usage (common in phishing)
                features['iframe_count'] = len(soup.find_all('iframe'))

                # Script usage
                features['script_count'] = len(soup.find_all('script'))

                # Link analysis
                links = soup.find_all('a')
                features['hyperlink_count'] = len(links)

                # External vs internal links
                domain = tldextract.extract(url).registered_domain
                external_links = 0
                for link in links:
                    if link.has_attr('href'):
                        link_domain = tldextract.extract(link['href']).registered_domain
                        if link_domain and link_domain != domain:
                            external_links += 1

                features['external_links_count'] = external_links
                features['external_links_ratio'] = external_links / len(links) if links else 0

                # Meta refresh redirect
                meta_refresh = soup.find('meta', {'http-equiv': 'refresh'})
                features['has_meta_refresh'] = 1 if meta_refresh else 0

                # Favicon from same domain
                favicon = soup.find('link', {'rel': 'icon'}) or soup.find('link', {'rel': 'shortcut icon'})
                features['favicon_same_domain'] = 0
                if favicon and favicon.has_attr('href'):
                    favicon_domain = tldextract.extract(favicon['href']).registered_domain
                    if favicon_domain == domain or not favicon_domain:  # No domain means relative URL
                        features['favicon_same_domain'] = 1

                # External resources (scripts, css, images)
                resources = soup.find_all(['script', 'link', 'img'])
                external_resources = 0
                for resource in resources:
                    if resource.has_attr('src'):
                        resource_domain = tldextract.extract(resource['src']).registered_domain
                        if resource_domain and resource_domain != domain:
                            external_resources += 1
                    elif resource.has_attr('href'):
                        resource_domain = tldextract.extract(resource['href']).registered_domain
                        if resource_domain and resource_domain != domain:
                            external_resources += 1

                features['external_resources_ratio'] = external_resources / len(resources) if resources else 0

                # Redirect count
                features['redirect_count'] = len(response.history)

            else:
                # Set default values if response is not 200
                features.update({
                    'title_length': 0,
                    'body_length': 0,
                    'form_count': 0,
                    'input_count': 0,
                    'iframe_count': 0,
                    'script_count': 0,
                    'hyperlink_count': 0,
                    'redirect_count': 0,
                    'favicon_same_domain': 0,
                    'external_resources_ratio': 0
                })

        except Exception as e:
            # Re-raise the exception to be handled by the caller
            raise e

        return features

    def _extract_whois_features(self, url, timeout=3):
        """Extract WHOIS-based features with timeout"""
        features = {}

        try:
            # Extract domain
            extracted = tldextract.extract(url)
            domain = f"{extracted.domain}.{extracted.suffix}"

            # Get WHOIS information
            w = whois.whois(domain)

            # Domain age
            creation_date = w.creation_date
            expiration_date = w.expiration_date

            if creation_date:
                if isinstance(creation_date, list):
                    creation_date = creation_date[0]

                domain_age = (datetime.now() - creation_date).days
                features['domain_age_days'] = domain_age
            else:
                features['domain_age_days'] = -1

            # Domain expiry
            if expiration_date:
                if isinstance(expiration_date, list):
                    expiration_date = expiration_date[0]

                domain_expiry = (expiration_date - datetime.now()).days
                features['domain_expiry_days'] = domain_expiry
            else:
                features['domain_expiry_days'] = -1

            # Privacy protection
            features['has_private_registration'] = 1 if "privacy" in str(w).lower() or "proxy" in str(w).lower() else 0

        except Exception as e:
            # Set default values on error
            features['domain_age_days'] = -1
            features['domain_expiry_days'] = -1
            features['has_private_registration'] = 0

            # Re-raise the exception to be handled by the caller
            raise e

        return features

    def predict(self, url):
        """Predict if a URL is a phishing website"""
        start_time = time.time()

        # Extract features from the URL
        features = {}
        features.update(self._extract_url_features(url))
        features.update(self._extract_domain_features(url))

        # Try to extract content features (with timeout)
        try:
            content_features = self._extract_content_features(url)
            features.update(content_features)
        except Exception:
            # Set default values if content extraction fails
            features.update({
                'title_length': 0,
                'body_length': 0,
                'form_count': 0,
                'input_count': 0,
                'iframe_count': 0,
                'script_count': 0,
                'hyperlink_count': 0,
                'redirect_count': 0,
                'favicon_same_domain': 0,
                'external_resources_ratio': 0
            })

        # Try to extract WHOIS features (with timeout)
        try:
            whois_features = self._extract_whois_features(url)
            features.update(whois_features)
        except Exception:
            # Set default values if whois extraction fails
            features.update({
                'domain_age_days': -1,
                'domain_expiry_days': -1,
                'has_private_registration': 0
            })

        # Convert features to DataFrame and ensure all feature columns exist
        features_df = pd.DataFrame([features])

        # Ensure all necessary columns exist with correct ordering
        for feature in self.feature_names:
            if feature not in features_df.columns:
                features_df[feature] = 0

        # Reorder columns to match training order
        features_df = features_df[self.feature_names]

        # Scale features
        features_scaled = self.scaler.transform(features_df)

        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        prediction_proba = self.model.predict_proba(features_scaled)[0]

        end_time = time.time()
        processing_time = end_time - start_time

        result = {
            'url': url,
            'is_phishing': bool(prediction),
            'confidence': float(prediction_proba[1] if prediction else prediction_proba[0]),
            'processing_time': processing_time
        }

        return result

    def load_model(self, model_path='phishing_detector_model.pkl', scaler_path='phishing_detector_scaler.pkl'):
        """Load the model and scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        # Load feature names
        with open('phishing_detector_features.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)

        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")
        print(f"Feature names loaded from phishing_detector_features.pkl")