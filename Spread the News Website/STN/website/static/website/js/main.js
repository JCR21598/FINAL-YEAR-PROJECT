
// JQuery Functionality - waits till page is loaded
$(document).ready(function(){

    // For the Detectfor form when it is submitted
    $("#detector-form").on("submit", function(event){

        // Prevent webpage from reloading
        event.preventDefault();

        // Call AJAX function
        detector_form();
    });
});


// AJAX function
function detector_form(){
    console.log($('#detector-field').val())
    $.ajax({
        url: "/input/",
        type: "POST",
        data: { input: $('#detector-field').val() },

        // Message before sending
       beforeSend: function(xhr, settings) {
            if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
                }
        },

        // If AJAX successful
        success: function(json){
            alert("Successful AJAX")
            $('#detector-field').val('');  // Reset Field
            console.log(json); // log the returned json to the console
        },

        // If AJAX unsuccessful
        error: function(){
            alert("AJAX Error")
        }
    });
}

// Provided by Django
function getCookie(name) {

    var cookieValue = null;
    if (document.cookie && document.cookie != '') {

        var cookies = document.cookie.split(';');

        for (var i = 0; i < cookies.length; i++) {

            var cookie = jQuery.trim(cookies[i]);

            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) == (name + '=')) {

                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

var csrftoken = getCookie('csrftoken');


 function csrfSafeMethod(method) {
    // these HTTP methods do not require CSRF protection
    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
}



//$(document).on('submit', '#detector-form', function(e) {
//
//
//    e.preventDefault(); //Meant to prevent from page to reload
//
//    $.ajax({
//        type: "POST",
//        url: "/input/",
//        data: {
//            input: $("#detector-input").val()
//        },
//        success: function(data){
//            alert(data)
//        }
//    })
//});